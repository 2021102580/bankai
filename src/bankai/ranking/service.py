
from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from bankai.candidates.generator import CandidateGenerator, GenerationResult
from bankai.config import Settings, get_settings
from bankai.domain.entities import Recommendation, RecurrenceProfile
from bankai.features.builder import FeatureContext, build_feature_context, build_feature_matrix
from bankai.features.embed import build_user_profile
from bankai.features.recurrence import detect_recurring
from bankai.logging_setup import get_logger
from bankai.models.registry import LoadedModel, ModelRegistry
from bankai.ranking.diversify import cap_by_category, dedupe_by_payee_and_due, mmr_rerank
from bankai.ranking.explain import explain
from bankai.ranking.policy import Decision, PolicyContext, PolicyEngine

log = get_logger(__name__)


class TransactionSource(Protocol):
    def transactions_for_user(self, user_id: str, *, since: dt.date | None = None) -> list: ...

    def account(self, user_id: str) -> dict | None: ...


@dataclass
class RequestContext:
    user_id: str
    as_of: dt.date
    horizon_days: int = 14
    limit: int = 5
    currency: str = "EUR"
    excluded: frozenset[str] = field(default_factory=frozenset)
    channel: str = "mobile_app"
    session_intent: str | None = None
    include_recently_paid: bool = True
    collect_trace: bool = False
    history_inclusive: bool = True


@dataclass
class CandidateTrace:

    counterparty_id: str
    counterparty_name: str
    category: str
    status: str
    source: str = "unknown"
    base_score: float = 0.0
    model_score: float = 0.0
    final_rank: int | None = None
    policy_decision: str = "allow"
    policy_flags: tuple[str, ...] = ()
    dropped_at: str | None = None
    drop_reason: str | None = None

    @property
    def was_returned(self) -> bool:
        return self.final_rank is not None


@dataclass
class RankingResult:
    recommendations: list[Recommendation]
    model_version: str
    degraded: bool
    degradation_reason: str | None
    diagnostics: dict[str, Any]
    latency_ms: float
    generated_at: dt.datetime
    trace: list[CandidateTrace] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.recommendations


class RankingService:
    def __init__(
        self,
        *,
        repository: TransactionSource,
        registry: ModelRegistry,
        settings: Settings | None = None,
    ) -> None:
        self.repo = repository
        self.registry = registry
        self.settings = settings or get_settings()
        self._policy = PolicyEngine(
            max_single=self.settings.max_single_transfer,
            daily_limit=self.settings.daily_transfer_limit,
            monthly_limit=self.settings.monthly_transfer_limit,
            balance_buffer=self.settings.insufficient_balance_buffer,
            cutoff_hour=self.settings.cutoff_hour_local,
            enable=self.settings.enable_policy,
            fail_open=self.settings.fail_open_on_budget,
        )
        self._generators: dict[int, CandidateGenerator] = {}
        self._ablation: dict[str, bool] = {}

    # ------------------------------------------------------------- ablation
    ABLATION_FLAGS: tuple[str, ...] = ("use_ranker", "use_policy", "use_mmr", "use_blend")

    def configure_ablation(
        self,
        *,
        use_ranker: bool = True,
        use_policy: bool = True,
        use_mmr: bool = True,
        use_blend: bool = True,
    ) -> None:
        requested = {
            "use_ranker": use_ranker,
            "use_policy": use_policy,
            "use_mmr": use_mmr,
            "use_blend": use_blend,
        }
        self._ablation = {k: v for k, v in requested.items() if not v}

    def _ablation_flag(self, name: str, default: bool = True) -> bool:
        return self._ablation.get(name, default)

    # ------------------------------------------------------------- internals
    def _generator_for(self, model: LoadedModel) -> CandidateGenerator:
        key = id(model)
        gen = self._generators.get(key)
        if gen is None:
            gen = CandidateGenerator(
                pool_size=self.settings.candidate_pool_size,
                encoder=model.encoder,
                latent=model.latent,
            )
            self._generators[key] = gen
        return gen

    def _build_context(self, profile, recs, model, ctx: RequestContext) -> FeatureContext:
        return build_feature_context(
            profile,
            recs,
            as_of=ctx.as_of,
            horizon_days=ctx.horizon_days,
            encoder=model.encoder,
            latent=model.latent,
            counterparty_ids=list(set(profile.series) | set(recs)),
        )

    def _blend(
        self, model: LoadedModel, features: np.ndarray
    ) -> tuple[np.ndarray, list[dict[str, float]], str]:
        ranker_scores = model.ranker.score(features)
        ranker_contrib = model.ranker.explain(features)
        used = "ranker"

        if not self._ablation_flag("use_ranker"):
            prior = model.fallback.score(features)
            return prior, model.fallback.explain(features), "heuristic_only"

        if self._ablation_flag("use_blend") and self.settings.enable_heuristic_blend:
            prior = model.fallback.score(features)
            w = float(np.clip(self.settings.heuristic_blend_weight, 0.0, 0.9))
            blended = (1.0 - w) * ranker_scores + w * prior
            for contribution in ranker_contrib:
                for name, value in list(contribution.items()):
                    contribution[name] = value * (1.0 - w)
            used = "ranker+heuristic"
            return blended, ranker_contrib, used

        return ranker_scores, ranker_contrib, used

    # ------------------------------------------------------------------ API
    def rank(self, ctx: RequestContext) -> RankingResult:
        started = time.perf_counter()
        generated_at = dt.datetime.now(tz=dt.UTC)
        model = self.registry.get()
        diagnostics: dict[str, Any] = {
            "model_version": model.version,
            "degraded": not model.healthy,
        }

        transactions = self.repo.transactions_for_user(ctx.user_id)
        profile = build_user_profile(
            ctx.user_id, transactions, as_of=ctx.as_of, inclusive=ctx.history_inclusive
        )
        recs: dict[str, RecurrenceProfile] = detect_recurring(
            profile.series, as_of=ctx.as_of, horizon_days=ctx.horizon_days
        )
        diagnostics["n_transactions"] = profile.n_transactions
        diagnostics["n_counterparties"] = profile.n_counterparties
        diagnostics["n_recurring"] = sum(1 for r in recs.values() if r.is_recurring)

        generator = self._generator_for(model)
        generation: GenerationResult = generator.generate(
            profile,
            recs,
            as_of=ctx.as_of,
            horizon_days=ctx.horizon_days,
            excluded=set(ctx.excluded),
        )
        candidates = list(generation.candidates)
        if not ctx.include_recently_paid:
            candidates = [c for c in candidates if c.status.value != "recently_paid"]
        diagnostics["recall"] = generation.recall_mix
        diagnostics["n_candidates"] = len(candidates)
        diagnostics["retrieved_ids"] = [c.counterparty_id for c in candidates]

        # ---- stage trace (opt-in) -----------------------------------------
        traces: dict[str, CandidateTrace] = {}
        if ctx.collect_trace:
            traces = {
                c.counterparty_id: CandidateTrace(
                    counterparty_id=c.counterparty_id,
                    counterparty_name=c.counterparty_name,
                    category=c.category,
                    status=c.status.value,
                    source=c.source,
                    base_score=c.base_score,
                )
                for c in candidates
            }

        def _empty(extra: dict | None = None) -> RankingResult:
            return RankingResult(
                recommendations=[],
                model_version=model.version,
                degraded=not model.healthy,
                degradation_reason=model.degradation_reason,
                diagnostics={**diagnostics, **(extra or {})},
                latency_ms=(time.perf_counter() - started) * 1000.0,
                generated_at=generated_at,
                trace=list(traces.values()),
            )

        if not candidates:
            return _empty({"coverage": generation.coverage_note})

        # ---- features + scoring ------------------------------------------
        feature_ctx = self._build_context(profile, recs, model, ctx)
        ids = [c.counterparty_id for c in candidates]
        features = build_feature_matrix(ids, feature_ctx)
        scores, contributions, used_model = self._blend(model, features)
        diagnostics["scorer"] = used_model

        order = np.argsort(-scores)
        ranked = [(candidates[i], float(scores[i]), contributions[i]) for i in order]
        for cand, score, _ in ranked:
            trace = traces.get(cand.counterparty_id)
            if trace is not None:
                trace.model_score = round(float(score), 6)

        # ---- policy --------------------------------------------------------
        policy_ctx = self._policy_context(profile, ctx, transactions)
        if not self._ablation_flag("use_policy"):
            allowed = [(c, s, c_, ()) for c, s, c_ in ranked]
            diagnostics["policy_suppressed"] = 0
        else:
            allowed = []
            suppressed = 0
            for cand, score, contrib in ranked:
                verdict = self._policy.evaluate(cand, policy_ctx)
                if verdict.decision is Decision.ALLOW:
                    allowed.append((cand, score, contrib, verdict.flags))
                    trace = traces.get(cand.counterparty_id)
                    if trace is not None:
                        trace.policy_flags = tuple(verdict.flags)
                else:
                    suppressed += 1
                    trace = traces.get(cand.counterparty_id)
                    if trace is not None:
                        trace.policy_decision = verdict.decision.value
                        trace.policy_flags = tuple(verdict.flags)
                        trace.dropped_at = "policy"
                        trace.drop_reason = verdict.reason
            diagnostics["policy_suppressed"] = suppressed

        if not allowed:
            return _empty({"coverage": "all_candidates_policy_blocked"})

        # ---- diversification ----------------------------------------------
        pre = [a[0] for a in allowed]
        pre_scores = [a[1] for a in allowed]
        keep = max(3, ctx.limit)  # over-fetch, then trim
        if self._ablation_flag("use_mmr"):
            idx = mmr_rerank(
                pre, pre_scores, k=min(keep, len(pre)), lam=self.settings.mmr_lambda, protected=2
            )
            idx = dedupe_by_payee_and_due(pre, idx)
        else:
            idx = list(range(min(keep, len(pre))))
        before_cap = list(idx)
        idx = cap_by_category(pre, idx, cap=4)
        idx = idx[: ctx.limit]

        if traces:
            shown = {pre[i].counterparty_id for i in idx}
            shortlisted = {pre[i].counterparty_id for i in before_cap}
            for cand, _s, _c, _flags in allowed:
                trace = traces.get(cand.counterparty_id)
                if trace is None or cand.counterparty_id in shown:
                    continue
                if cand.counterparty_id in shortlisted:
                    trace.dropped_at = "limit"
                    trace.drop_reason = "outside the requested result limit"
                else:
                    trace.dropped_at = "diversification"
                    trace.drop_reason = "dropped by category cap or near-duplicate rule"

        # ---- assemble -------------------------------------------------------
        recommendations: list[Recommendation] = []
        for rank, i in enumerate(idx, start=1):
            cand, score, contrib, flags = allowed[i]
            rec = cand.recurrence
            confidence = float(np.clip(rec.confidence if rec else 0.35, 0.0, 1.0))
            recommendations.append(
                Recommendation(
                    rank=rank,
                    candidate=cand,
                    score=round(float(score), 6),
                    confidence=round(confidence, 4),
                    contributions={
                        k: round(v, 5)
                        for k, v in sorted(contrib.items(), key=lambda kv: -abs(kv[1]))[:8]
                    },
                    explanations=explain(cand, contributions=contrib, as_of=ctx.as_of, top_n=3),
                    policy_flags=list(flags),
                )
            )

        for rec in recommendations:
            trace = traces.get(rec.candidate.counterparty_id)
            if trace is not None:
                trace.final_rank = rec.rank
                trace.policy_flags = tuple(rec.policy_flags)
        diagnostics["returned"] = len(recommendations)
        return RankingResult(
            recommendations=recommendations,
            model_version=model.version,
            degraded=not model.healthy,
            degradation_reason=model.degradation_reason,
            diagnostics=diagnostics,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            generated_at=generated_at,
            trace=list(traces.values()),
        )

    def _policy_context(
        self, profile, ctx: RequestContext, transactions: list | None = None
    ) -> PolicyContext:
        account: dict | None = None
        try:
            account = self.repo.account(ctx.user_id)
        except Exception:  # pragma: no cover - account service down
            log.warning("account lookup failed; falling back to ledger balance")

        balance = None
        blocked: frozenset[str] = frozenset()
        if account:
            balance = account.get("available_balance")
            blocked = frozenset(account.get("blocked_counterparties") or ())
        if balance is None:
            balance = _ledger_balance(transactions, ctx.as_of, inclusive=ctx.history_inclusive)
        if balance is None:
            balance = profile.available_balance(fallback=0.0)

        cutoff_passed = (
            ctx.as_of >= dt.date.today()
            and dt.datetime.now().hour >= self.settings.cutoff_hour_local
        )
        return PolicyContext(
            user_id=ctx.user_id,
            as_of=ctx.as_of,
            available_balance=float(balance or 0.0),
            currency=account.get("currency", ctx.currency) if account else ctx.currency,
            blocked_counterparties=blocked,
            daily_limit=float(
                (account or {}).get("daily_limit") or self.settings.daily_transfer_limit
            ),
            monthly_limit=float(
                (account or {}).get("monthly_limit") or self.settings.monthly_transfer_limit
            ),
            cutoff_hour=self.settings.cutoff_hour_local,
            is_cutoff_passed=cutoff_passed,
        )

    # ------------------------------------------------------------ diagnostics
    def recurring_summary(
        self,
        user_id: str,
        as_of: dt.date | None = None,
        *,
        inclusive: bool = True,
    ) -> list[dict]:
        transactions = self.repo.transactions_for_user(user_id)
        if not transactions:
            return []
        as_of = as_of or max(t.value_date for t in transactions)
        profile = build_user_profile(user_id, transactions, as_of=as_of, inclusive=inclusive)
        recs = detect_recurring(profile.series, as_of=as_of, horizon_days=14)
        rows = [r for r in recs.values() if r.is_recurring]
        rows.sort(key=lambda r: r.days_until_due if r.days_until_due is not None else 999)
        return [
            {
                "counterparty_id": r.counterparty_id,
                "counterparty_name": r.counterparty_name,
                "category": r.category,
                "period_days": r.period_days,
                "confidence": round(r.confidence, 4),
                "adherence": round(r.adherence, 4),
                "expected_amount": r.expected_amount,
                "last_paid_on": r.last_paid_on,
                "next_due_on": r.next_due_on,
                "status": r.status.value,
                "observations": r.n_observations,
            }
            for r in rows
        ]


def _ledger_balance(
    transactions: list | None, as_of: dt.date, *, inclusive: bool = True
) -> float | None:
    if not transactions:
        return None
    cutoff = as_of.toordinal() + (1 if inclusive else 0)
    known = [
        (t.value_date.toordinal(), t.balance_after)
        for t in transactions
        if t.balance_after is not None and t.value_date.toordinal() < cutoff
    ]
    if not known:
        return None
    return float(max(known)[1])


__all__ = ["RankingResult", "RankingService", "RequestContext"]
