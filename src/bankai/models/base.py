

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from bankai.features.builder import N_FEATURES


@runtime_checkable
class Scorer(Protocol):
    name: str

    def score(self, features: np.ndarray) -> np.ndarray:
        """Return one score per row. Higher is better; scale is model-specific."""
        ...

    def explain(self, features: np.ndarray) -> list[dict[str, float]]:
        """Per-row feature contributions, signed, summing to the raw score."""
        ...

    @property
    def n_features(self) -> int: ...


class BaseScorer:
    name = "base"
    n_features = N_FEATURES

    def score(self, features: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def explain(self, features: np.ndarray) -> list[dict[str, float]]:  # pragma: no cover
        raise NotImplementedError


def normalise(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    lo, hi = float(np.min(values)), float(np.max(values))
    if hi - lo < 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - lo) / (hi - lo)


def rank_normalise(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    order = np.argsort(np.argsort(values))
    return order / max(1.0, values.size - 1.0)
