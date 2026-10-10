

from __future__ import annotations

import datetime as dt
import json
import time
from dataclasses import dataclass, field

from bankai.config import Settings, get_settings
from bankai.data.store import SqliteTransactionRepository
from bankai.evaluation.backtest import build_cases, run_ablation, run_backtest
from bankai.logging_setup import configure_logging, get_logger
from bankai.models.registry import ModelRegistry
from bankai.ranking.service import RankingService, RequestContext
from bankai.training.trainer import fit_auxiliary_models, train_and_publish

log = get_logger(__name__)


@dataclass
class PipelineResult:
    as_of: dt.date
    n_users: int
    n_transactions: int
    training: dict = field(default_factory=dict)
    backtest: dict = field(default_factory=dict)
    ablation: dict = field(default_factory=dict)
    duration_s: float = 0.0
    example: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "as_of": self.as_of.isoformat(),
            "n_users": self.n_users,
            "n_transactions": self.n_transactions,
            "training": self.training,
            "backtest": self.backtest,
            "ablation": self.ablation,
            "duration_s": round(self.duration_s, 2),
            "example": self.example,
        }


def load_corpus(settings: Settings) -> tuple[dict[str, list], SqliteTransactionRepository, dt.date]:
    from bankai.data.synthetic import generate_dataset

    repo = SqliteTransactionRepository(settings.db_path)
    if repo.count() == 0:
        as_of = (
            dt.date.fromisoformat(settings.as_of)
            if settings.as_of
            else dt.date.today() - dt.timedelta(days=1)
        )
        log.info(
            "generating synthetic corpus",
            extra={"users": settings.n_users, "months": settings.n_months},
        )
        transactions, _, snapshot = generate_dataset(
            n_users=settings.n_users,
            months=settings.n_months,
            as_of=as_of,
            seed=settings.seed,
        )
        repo.bulk_insert(transactions)
        by_user: dict[str, list] = {}
        for t in transactions:
            by_user.setdefault(t.user_id, []).append(t)
        for user_id, _txs in by_user.items():
            repo.upsert_account(
                user_id,
                available_balance=None,
                currency="EUR",
                daily_limit=settings.daily_transfer_limit,
                monthly_limit=settings.monthly_transfer_limit,
                blocked_counterparties=[],
            )
        return by_user, repo, snapshot

    transactions = repo.all_transactions()
    by_user: dict[str, list] = {}
    for t in transactions:
        by_user.setdefault(t.user_id, []).append(t)
    snapshot = repo.latest_date() or dt.date.today()
    return by_user, repo, snapshot


def run_pipeline(
    *,
    settings: Settings | None = None,
    max_users: int | None = None,
    backtest_users: int = 40,
    force_promote: bool = True,
) -> PipelineResult:
    started = time.perf_counter()
    s = settings or get_settings()
    configure_logging(s.log_level, s.log_format)

    by_user, repo, as_of = load_corpus(s)
    train_users = max_users or s.n_users
    log.info("corpus ready", extra={"users": len(by_user), "as_of": as_of.isoformat()})

    encoder, latent = fit_auxiliary_models(by_user)
    report = train_and_publish(
        by_user,
        as_of=as_of,
        settings=s,
        encoder=encoder,
        latent=latent,
        max_users=train_users,
        force_promote=force_promote,
    )
    log.info("training complete", extra=report.to_dict())

    # ---- backtest on held-out decision points ------------------------------
    registry = ModelRegistry(s.artifact_dir)
    service = RankingService(repository=repo, registry=registry, settings=s)
    subset = dict(sorted(by_user.items())[:backtest_users])
    cases = build_cases(subset, as_of=as_of, label_window_days=s.label_window_days)
    log.info("backtest cases", extra={"cases": len(cases), "users": len(subset)})

    result = run_backtest(
        service,
        cases,
        label="ranker_blended",
        limit=s.default_recommendations,
        horizon_days=s.default_horizon_days,
    )
    ablation = run_ablation(
        service, cases, limit=s.default_recommendations, horizon_days=s.default_horizon_days
    )

    example = _example(service, sorted(by_user)[0], as_of)
    return PipelineResult(
        as_of=as_of,
        n_users=len(by_user),
        n_transactions=sum(len(v) for v in by_user.values()),
        training=report.to_dict(),
        backtest=result.to_dict(),
        ablation={k: v.to_dict() for k, v in ablation.items()},
        duration_s=time.perf_counter() - started,
        example=example,
    )


def _example(service: RankingService, user_id: str, as_of: dt.date) -> dict:
    res = service.rank(RequestContext(user_id=user_id, as_of=as_of, horizon_days=14, limit=5))
    return {
        "user_id": user_id,
        "latency_ms": round(res.latency_ms, 2),
        "items": [
            {
                "rank": r.rank,
                "payee": r.candidate.counterparty_name,
                "category": r.candidate.category,
                "amount": r.candidate.suggested_amount,
                "currency": r.candidate.currency,
                "scheduled_for": r.candidate.scheduled_for.isoformat(),
                "status": r.candidate.status.value,
                "score": r.score,
                "confidence": r.confidence,
                "why": [e.text for e in r.explanations][:2],
            }
            for r in res.recommendations
        ],
    }


def main() -> int:  # pragma: no cover - CLI entry
    import argparse

    parser = argparse.ArgumentParser(description="Run the BankAI offline pipeline")
    parser.add_argument("--users", type=int, default=None)
    parser.add_argument("--backtest-users", type=int, default=40)
    parser.add_argument("--as-of", type=str, default=None)
    parser.add_argument("--no-promote", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    if args.as_of:
        settings.as_of = args.as_of
    result = run_pipeline(
        settings=settings,
        max_users=args.users,
        backtest_users=args.backtest_users,
        force_promote=not args.no_promote,
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
