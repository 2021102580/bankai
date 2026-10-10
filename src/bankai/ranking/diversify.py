

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence

import numpy as np

from bankai.domain.entities import TransferCandidate


def _similarity(a: TransferCandidate, b: TransferCandidate) -> float:
    """Blend structural and categorical similarity into one [0, 1] distance."""
    same_category = 1.0 if a.category == b.category else 0.0
    if a.scheduled_for == b.scheduled_for:
        date_closeness = 1.0
    else:
        gap = abs((a.scheduled_for - b.scheduled_for).days)
        date_closeness = float(np.exp(-gap / 5.0))
    amount_ratio = (
        min(a.suggested_amount, b.suggested_amount) / max(a.suggested_amount, b.suggested_amount)
        if max(a.suggested_amount, b.suggested_amount) > 0
        else 1.0
    )
    return float(
        np.clip(0.45 * same_category + 0.35 * date_closeness + 0.20 * amount_ratio, 0.0, 1.0)
    )


def mmr_rerank(
    candidates: Sequence[TransferCandidate],
    scores: Sequence[float],
    *,
    k: int = 5,
    lam: float = 0.35,
    protected: int = 0,
) -> list[int]:
    if not candidates or k <= 0:
        return []
    if len(candidates) <= k:
        return list(range(len(candidates)))

    scores_arr = np.asarray(scores, dtype=float)
    lo, hi = float(scores_arr.min()), float(scores_arr.max())
    if hi - lo < 1e-12:
        rel = np.zeros_like(scores_arr)
    else:
        rel = (scores_arr - lo) / (hi - lo)

    remaining = set(range(len(candidates)))
    selected: list[int] = list(range(min(protected, len(candidates))))
    remaining -= set(selected)

    while len(selected) < k and remaining:
        best_idx, best_value = None, -np.inf
        for idx in remaining:
            penalty = max(
                (_similarity(candidates[idx], candidates[j]) for j in selected), default=0.0
            )
            value = (1.0 - lam) * rel[idx] - lam * penalty
            if value > best_value:
                best_idx, best_value = idx, value
        if best_idx is None:  # pragma: no cover - remaining is non-empty
            break
        selected.append(best_idx)
        remaining.discard(best_idx)
    return selected


def cap_by_category(
    candidates: Sequence[TransferCandidate], indices: Sequence[int], cap: int
) -> list[int]:
    seen: dict[str, int] = {}
    out: list[int] = []
    for idx in indices:
        cat = candidates[idx].category
        if seen.get(cat, 0) >= cap:
            continue
        seen[cat] = seen.get(cat, 0) + 1
        out.append(idx)
    return out


def dedupe_by_payee_and_due(
    candidates: Sequence[TransferCandidate], indices: Sequence[int]
) -> list[int]:
    seen: set[tuple[str, dt.date]] = set()
    out: list[int] = []
    for idx in indices:
        key = (candidates[idx].counterparty_id, candidates[idx].scheduled_for)
        if key in seen:
            continue
        seen.add(key)
        out.append(idx)
    return out


__all__ = ["cap_by_category", "dedupe_by_payee_and_due", "mmr_rerank"]
