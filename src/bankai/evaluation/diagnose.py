

from __future__ import annotations

import datetime as dt
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from bankai.evaluation.backtest import BacktestCase
from bankai.evaluation.metrics import evaluate_rankings
from bankai.logging_setup import get_logger
from bankai.ranking.service import RankingService, RequestContext

log = get_logger(__name__)

MISSED = "missed"
RETURNED = "returned"


@dataclass
class CaseDiagnosis:
    user_id: str
    as_of: dt.date
    n_relevant: int
    relevant: set[str] = field(default_factory=set)
    # counterparty_id -> how it failed
    failures: dict[str, str] = field(default_factory=dict)
    ranking: list[str] = field(default_factory=list)
    pool_size: int = 0
    latency_ms: float = 0.0

    @property
    def returned_all(self) -> bool:
        return self.relevant.issubset(set(self.ranking))

    @property
    def stages(self) -> Counter:
        return Counter(self.failures.values())


@dataclass
class DiagnosisReport:
    n_cases: int = 0
    n_relevant: int = 0
    n_returned: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    #: failure stage -> number of *relevant* payees lost there
    stage_losses: Counter = field(default_factory=Counter)
    #: stage -> share of all losses
    stage_share: dict[str, float] = field(default_factory=dict)
    #: Hit@k conditioned on the failure stage being clean
    by_stage_but_clean: dict[str, float] = field(default_factory=dict)
    #: quality of cases, by number of simultaneous targets
    by_n_relevant: dict[int, float] = field(default_factory=dict)
    #: user history length bucket -> Hit@5
    by_history: dict[str, float] = field(default_factory=dict)
    #: target position in the candidate pool when it was missed
    example_misses: list[dict[str, Any]] = field(default_factory=list)
    cases: list[CaseDiagnosis] = field(default_factory=list)

    @property
    def total_losses(self) -> int:
        return sum(self.stage_losses.values())

    @property
    def recall_rate(self) -> float:
        return 1.0 - (self.stage_losses["recall"] / self.total_losses) if self.total_losses else 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_cases": self.n_cases,
            "n_relevant": self.n_relevant,
            "n_returned": self.n_returned,
            "metrics": {k: round(v, 5) for k, v in self.metrics.items()},
            "stage_losses": dict(self.stage_losses),
            "stage_share": {k: round(v, 4) for k, v in self.stage_share.items()},
            "hit@5_by_history_bucket": {k: round(v, 4) for k, v in self.by_history.items()},
            "hit@5_by_n_relevant": {str(k): round(v, 4) for k, v in self.by_n_relevant.items()},
            "n_losses": self.total_losses,
        }


def _history_bucket(n_transactions: int) -> str:
    if n_transactions < 40:
        return "<40"
    if n_transactions < 100:
        return "40-99"
    if n_transactions < 200:
        return "100-199"
    return "200+"


def diagnose(
    service: RankingService,
    cases: Sequence[BacktestCase],
    *,
    limit: int = 5,
    horizon_days: int = 14,
    max_examples: int = 12,
) -> DiagnosisReport:
    report = DiagnosisReport()
    rankings: list[list[tuple[str, float]]] = []
    relevants: list[set[str]] = []
    by_history: dict[str, list[int]] = defaultdict(list)
    by_n_relevant: dict[int, list[int]] = defaultdict(list)
    examples: list[dict[str, Any]] = []

    for case in cases:
        ctx = RequestContext(
            user_id=case.user_id,
            as_of=case.as_of,
            horizon_days=horizon_days,
            limit=limit,
            collect_trace=True,
            history_inclusive=False,
        )
        result = service.rank(ctx)

        diagnosis = CaseDiagnosis(
            user_id=case.user_id,
            as_of=case.as_of,
            n_relevant=len(case.relevant),
            relevant=set(case.relevant),
            pool_size=len(result.trace),
            latency_ms=result.latency_ms,
        )
        trace_by_id = {t.counterparty_id: t for t in result.trace}

        for rec in result.recommendations:
            diagnosis.ranking.append(rec.candidate.counterparty_id)
        returned = set(diagnosis.ranking)
        hit = 1 if case.relevant <= returned else 0

        for cp_id in case.relevant:
            if cp_id in returned:
                report.stage_losses["returned"] += 0
                continue
            entry = trace_by_id.get(cp_id)
            if entry is None:
                stage = "recall"
                reason = "never generated as a candidate"
            elif entry.dropped_at == "policy":
                stage = "policy"
                reason = f"blocked by policy: {entry.policy_flags or entry.drop_reason}"
            elif entry.dropped_at in {"diversification", "limit"}:
                stage = "diversification"
                reason = f"trimmed at {entry.dropped_at}"
            else:
                stage = "ranking"
                reason = f"scored {entry.model_score:.3f}, below the cut-off"
            diagnosis.failures[cp_id] = stage
            report.stage_losses[stage] += 1
            if len(examples) < max_examples:
                examples.append(
                    {
                        "user_id": case.user_id,
                        "as_of": case.as_of.isoformat(),
                        "counterparty_id": cp_id,
                        "stage": stage,
                        "reason": reason,
                        "name": (entry.counterparty_name if entry else "?"),
                    }
                )

        report.n_relevant += len(case.relevant)
        report.n_returned += len(case.relevant & returned)
        report.n_cases += 1
        report.cases.append(diagnosis)
        rankings.append([(cp, 1.0 - i * 1e-3) for i, cp in enumerate(diagnosis.ranking)])
        relevants.append(case.relevant)

        n_tx = result.diagnostics.get("n_transactions", 0)
        by_history[_history_bucket(n_tx)].append(hit)
        by_n_relevant[len(case.relevant)].append(hit)

    report.metrics = evaluate_rankings(rankings, relevants) if rankings else {}
    total = report.total_losses
    report.stage_share = {
        stage: (count / total if total else 0.0) for stage, count in report.stage_losses.items()
    }
    report.by_history = {k: float(sum(v) / len(v)) for k, v in sorted(by_history.items())}
    report.by_n_relevant = {k: float(sum(v) / len(v)) for k, v in sorted(by_n_relevant.items())}
    report.example_misses = examples
    return report


def render(report: DiagnosisReport) -> str:
    m = report.metrics
    lines = [
        "",
        "=" * 74,
        "FAILURE ATTRIBUTION",
        "=" * 74,
        f"  cases={report.n_cases}  relevant targets={report.n_relevant}  "
        f"returned={report.n_returned}  lost={report.total_losses}",
        f"  hit@1={m.get('hit@1', 0):.4f}  hit@3={m.get('hit@3', 0):.4f}  "
        f"hit@5={m.get('hit@5', 0):.4f}  ndcg@5={m.get('ndcg@5', 0):.4f}",
        "",
        "  Where the targets are lost:",
    ]
    for stage in ("recall", "ranking", "policy", "diversification"):
        count = report.stage_losses.get(stage, 0)
        share = report.stage_share.get(stage, 0.0)
        bar = "#" * round(share * 40)
        lines.append(f"    {stage:16s} {count:5d}  {share:6.1%}  {bar}")
    lines += ["", "  Hit@5 by user history length:"]
    for bucket, value in report.by_history.items():
        lines.append(f"    {bucket:>8s} tx  {value:6.1%}")
    lines += ["", "  Hit@5 by number of simultaneous targets:"]
    for n, value in report.by_n_relevant.items():
        lines.append(f"    {n} target(s)      {value:6.1%}")
    if report.example_misses:
        lines += ["", "  Example misses:"]
        for ex in report.example_misses[:8]:
            lines.append(
                f"    [{ex['stage']:15s}] {ex['name'][:26]:26s} {ex['as_of']}  {ex['reason']}"
            )
    lines.append("")
    return "\n".join(lines)


__all__ = ["CaseDiagnosis", "DiagnosisReport", "diagnose", "render"]
