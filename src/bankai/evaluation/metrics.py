

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

NEG_INF = -1e9


def dcg(gains: Sequence[float]) -> float:
    return float(sum(g / math.log2(i + 2) for i, g in enumerate(gains)))


def ndcg_at_k(ranked: Sequence[tuple[str, float]], relevant: set[str], k: int = 5) -> float:
    if not relevant:
        return 0.0
    gains = [1.0 if item_id in relevant else 0.0 for item_id, _ in ranked[:k]]
    ideal = [1.0] * min(k, len(relevant))
    idcg = dcg(ideal)
    return dcg(gains) / idcg if idcg > 0 else 0.0


def hit_rate_at_k(ranked: Sequence[tuple[str, float]], relevant: set[str], k: int = 5) -> float:
    return 1.0 if any(item_id in relevant for item_id, _ in ranked[:k]) else 0.0


def precision_at_k(ranked: Sequence[tuple[str, float]], relevant: set[str], k: int = 5) -> float:
    if not ranked[:k]:
        return 0.0
    return sum(1 for item_id, _ in ranked[:k] if item_id in relevant) / min(k, len(ranked))


def mrr(ranked: Sequence[tuple[str, float]], relevant: set[str], k: int = 10) -> float:
    for idx, (item_id, _) in enumerate(ranked[:k], start=1):
        if item_id in relevant:
            return 1.0 / idx
    return 0.0


def recall_at_k(ranked: Sequence[tuple[str, float]], relevant: set[str], k: int = 5) -> float:
    if not relevant:
        return 0.0
    found = sum(1 for item_id, _ in ranked[:k] if item_id in relevant)
    return found / len(relevant)


def evaluate_rankings(
    rankings: Sequence[Sequence[tuple[str, float]]],
    relevants: Sequence[set[str]],
    ks: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, float]:
    if not rankings or len(rankings) != len(relevants):
        raise ValueError("rankings and relevants must be non-empty and the same length")
    out: dict[str, float] = {"n_groups": float(len(rankings))}
    for k in ks:
        out[f"ndcg@{k}"] = float(
            np.mean([ndcg_at_k(r, rel, k) for r, rel in zip(rankings, relevants, strict=True)])
        )
        out[f"hit@{k}"] = float(
            np.mean([hit_rate_at_k(r, rel, k) for r, rel in zip(rankings, relevants, strict=True)])
        )
        out[f"precision@{k}"] = float(
            np.mean([precision_at_k(r, rel, k) for r, rel in zip(rankings, relevants, strict=True)])
        )
        out[f"recall@{k}"] = float(
            np.mean([recall_at_k(r, rel, k) for r, rel in zip(rankings, relevants, strict=True)])
        )
    out["mrr"] = float(np.mean([mrr(r, rel) for r, rel in zip(rankings, relevants, strict=True)]))
    return out


def bootstrap_ci(
    values: Sequence[float], *, n_resamples: int = 500, alpha: float = 0.05, seed: int = 13
) -> tuple[float, float, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    point = float(arr.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    means = arr[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return point, lo, hi


def compare(
    baseline: Sequence[Sequence[tuple[str, float]]],
    candidate: Sequence[Sequence[tuple[str, float]]],
    relevants: Sequence[set[str]],
    k: int = 5,
    n_resamples: int = 500,
    seed: int = 17,
) -> dict[str, float]:
    base = np.asarray([ndcg_at_k(r, rel, k) for r, rel in zip(baseline, relevants, strict=True)])
    cand = np.asarray([ndcg_at_k(r, rel, k) for r, rel in zip(candidate, relevants, strict=True)])
    diff = cand - base
    point, lo, hi = bootstrap_ci(diff, n_resamples=n_resamples, seed=seed)
    n = diff.size
    se = float(diff.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    return {
        "ndcg_gain": point,
        "ci_low": lo,
        "ci_high": hi,
        "std_error": se,
        "significant": bool(lo > 0.0 or hi < 0.0),
        "wins": float((diff > 0).sum()),
        "losses": float((diff < 0).sum()),
        "ties": float((diff == 0).sum()),
    }


def coverage(generated: Sequence[int], relevant_total: Sequence[int]) -> float:
    num = float(np.sum(generated))
    den = float(np.sum(relevant_total))
    return num / den if den > 0 else 0.0


__all__ = [
    "bootstrap_ci",
    "compare",
    "coverage",
    "dcg",
    "evaluate_rankings",
    "hit_rate_at_k",
    "mrr",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
]
