

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from bankai.evaluation.metrics import compare, evaluate_rankings
from bankai.logging_setup import get_logger
from bankai.ranking.service import RankingService, RequestContext

log = get_logger(__name__)


@dataclass
class BacktestCase:
    user_id: str
    as_of: dt.date
    relevant: set[str]
    pool_size: int
    retrieved: set[str] = field(default_factory=set)
    ranking: list[tuple[str, float]] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


@dataclass
class BacktestResult:
    name: str
    metrics: dict[str, float] = field(default_factory=dict)
    n_cases: int = 0
    coverage: float = 0.0
    empty_rate: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    rankings: list[list[tuple[str, float]]] = field(default_factory=list)
    relevants: list[set[str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_cases": self.n_cases,
            "metrics": {k: round(v, 5) for k, v in self.metrics.items()},
            "coverage": round(self.coverage, 5),
            "empty_rate": round(self.empty_rate, 5),
            "latency_p50_ms": round(self.latency_p50_ms, 2),
            "latency_p95_ms": round(self.latency_p95_ms, 2),
        }


def build_cases(
    transactions_by_user: dict[str, list],
    *,
    as_of: dt.date,
    label_window_days: int = 3,
    min_gap_days: int = 3,
    min_history_days: int = 45,
    max_cases_per_user: int = 12,
) -> list[BacktestCase]:
    from bankai.training.labeler import find_decision_days

    cases: list[BacktestCase] = []
    for user_id, txs in sorted(transactions_by_user.items()):
        days = find_decision_days(
            txs,
            as_of=as_of,
            min_gap_days=min_gap_days,
            horizon=label_window_days,
        )
        days = [d for d in days if (as_of - d).days >= min_history_days][-max_cases_per_user:]
        for day in days:
            window_end = day + dt.timedelta(days=label_window_days)
            relevant = {
                t.counterparty_id
                for t in txs
                if t.direction.value == "outbound"
                and t.status.value == "posted"
                and day <= t.value_date <= window_end
            }
            if not relevant:
                continue
            cases.append(
                BacktestCase(user_id=user_id, as_of=day, relevant=relevant, pool_size=len(relevant))
            )
    return cases


def run_backtest(
    service: RankingService,
    cases: list[BacktestCase],
    *,
    label: str = "production",
    limit: int = 5,
    horizon_days: int = 14,
    on_progress: Callable[[int, int], None] | None = None,
) -> BacktestResult:
    rankings: list[list[tuple[str, float]]] = []
    relevants: list[set[str]] = []
    latencies: list[float] = []
    retrieved_total = relevant_total = 0
    empty = 0

    for i, case in enumerate(cases):
        ctx = RequestContext(
            user_id=case.user_id,
            as_of=case.as_of,
            horizon_days=horizon_days,
            limit=limit,
            history_inclusive=False,
        )
        try:
            result = service.rank(ctx)
        except Exception as exc:  # pragma: no cover - a backtest must not abort on one bad case
            log.exception(
                "backtest case failed", extra={"user_id": case.user_id, "error": str(exc)}
            )
            continue
        ranked = [(r.candidate.counterparty_id, r.score) for r in result.recommendations]
        case.ranking = ranked
        case.recommendations = [r.candidate.counterparty_id for r in result.recommendations]
        pool = set(result.diagnostics.get("retrieved_ids", [])) or set(case.recommendations)
        case.retrieved = pool
        retrieved_total += len(pool & case.relevant)
        relevant_total += len(case.relevant)
        if not ranked:
            empty += 1
        rankings.append(ranked)
        relevants.append(case.relevant)
        latencies.append(result.latency_ms)
        if on_progress and (i + 1) % 100 == 0:
            on_progress(i + 1, len(cases))

    metrics = evaluate_rankings(rankings, relevants) if rankings else {}
    return BacktestResult(
        name=label,
        metrics=metrics,
        n_cases=len(rankings),
        coverage=retrieved_total / relevant_total if relevant_total else 0.0,
        empty_rate=empty / len(cases) if cases else 0.0,
        latency_p50_ms=float(np.percentile(latencies, 50)) if latencies else 0.0,
        latency_p95_ms=float(np.percentile(latencies, 95)) if latencies else 0.0,
        rankings=rankings,
        relevants=relevants,
    )


def run_ablation(
    service: RankingService,
    cases: list[BacktestCase],
    *,
    variants: dict[str, dict[str, Any]] | None = None,
    limit: int = 5,
    horizon_days: int = 14,
) -> dict[str, BacktestResult]:
    variants = variants or {
        "heuristic_only": {"use_ranker": False, "use_policy": True, "use_mmr": True},
        "ranker_only": {"use_ranker": True, "use_policy": True, "use_mmr": True},
        "ranker_no_policy": {"use_ranker": True, "use_policy": False, "use_mmr": True},
        "ranker_no_mmr": {"use_ranker": True, "use_policy": True, "use_mmr": False},
    }
    results: dict[str, BacktestResult] = {}
    for name, overrides in variants.items():
        service.configure_ablation(**overrides)
        results[name] = run_backtest(
            service, cases, label=name, limit=limit, horizon_days=horizon_days
        )
    service.configure_ablation()  # restore production config
    return results


def significance(
    a: list[list[tuple[str, float]]],
    b: list[list[tuple[str, float]]],
    relevants: list[set[str]],
    *,
    k: int = 5,
) -> dict[str, float]:
    return compare(a, b, relevants, k=k)


__all__ = [
    "BacktestCase",
    "BacktestResult",
    "build_cases",
    "run_ablation",
    "run_backtest",
    "significance",
]
