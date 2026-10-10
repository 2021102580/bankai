
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from bankai.features.builder import FEATURE_NAMES, N_FEATURES
from bankai.models.base import BaseScorer, normalise

IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}


@dataclass
class HeuristicWeights:
    """Weights are interpretable and hand-tuned against offline backtests."""

    overdue: float = 2.4
    due_now: float = 2.2
    due_soon: float = 1.7
    recently_paid: float = -0.6
    recurring: float = 0.15
    one_off: float = -0.8
    confidence: float = 1.9
    adherence: float = 0.8
    amount_stability: float = 0.7
    schedule_stability: float = 0.9
    share_of_monthly_spend: float = -0.55
    amount_vs_user_median: float = -0.35
    spend_share: float = -0.25
    novelty: float = -0.45
    success_rate: float = 0.3
    failed_recent: float = -0.8
    missed_cycles: float = -0.7
    observations_log: float = 0.6
    decay_half_life_days: float = 9.0
    status_weights: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        data = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        return data

    @classmethod
    def from_dict(cls, data: dict) -> HeuristicWeights:
        known = {k: v for k, v in data.items() if k in cls.__annotations__}
        return cls(**known)


class HeuristicScorer(BaseScorer):
    name = "heuristic"

    def __init__(self, weights: HeuristicWeights | None = None) -> None:
        self.w = weights or HeuristicWeights()
        self.n_features = N_FEATURES
        self._status_bonus = {
            "overdue": self.w.overdue,
            "due_now": self.w.due_now,
            "due_soon": self.w.due_soon,
            "recently_paid": self.w.recently_paid,
            "recurring": self.w.recurring,
            "one_off": self.w.one_off,
            "unknown": 0.0,
        }

    # ------------------------------------------------------------------ score
    def raw_scores(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return np.zeros(0, dtype=float)
        g = lambda name: x[:, IDX[name]]  # noqa: E731

        score = np.zeros(x.shape[0], dtype=float)
        score += 2.2 * g("is_overdue")
        score += 2.0 * g("is_due_now")
        score += 1.5 * g("is_due_soon")
        score -= 0.6 * g("is_recently_paid")
        score += self.w.confidence * g("recurrence_confidence")
        score += self.w.schedule_stability * g("schedule_stability")
        score += self.w.amount_stability * g("amount_is_fixed")
        score += self.w.adherence * g("adherence")
        score += self.w.observations_log * g("observations_log")
        score -= self.w.missed_cycles * g("missed_cycles")
        score -= self.w.share_of_monthly_spend * g("share_of_monthly_spend")
        score -= self.w.amount_vs_user_median * np.clip(g("amount_vs_user_median"), 0, 10)
        score -= self.w.spend_share * g("spend_share")
        score -= self.w.novelty * g("novelty")
        score += self.w.success_rate * g("success_rate")
        score -= self.w.failed_recent * g("failed_recent")

        days_until = np.clip(g("days_until_due"), 0.0, 120.0)
        score += (
            0.9
            * np.exp(-days_until / max(self.w.decay_half_life_days, 1e-6))
            * g("horizon_proximity")
        )
        score += 0.6 * g("periodicity_confidence") * g("is_recurring")
        return score

    def score(self, features: np.ndarray) -> np.ndarray:
        return normalise(self.raw_scores(features))

    # -------------------------------------------------------------- explain
    #: ``(feature_name, contribution_name)``; the coefficient carries the sign.
    _TERMS = (
        ("is_overdue", "overdue"),
        ("is_due_now", "due_now"),
        ("is_due_soon", "due_soon"),
        ("recurrence_confidence", "recurrence_confidence"),
        ("schedule_stability", "schedule_stability"),
        ("adherence", "adherence"),
        ("amount_is_fixed", "amount_stability"),
        ("missed_cycles", "missed_cycles"),
        ("share_of_monthly_spend", "share_of_monthly_spend"),
        ("amount_vs_user_median", "amount_vs_user_median"),
        ("novelty", "novelty"),
        ("spend_share", "spend_share"),
        ("is_recently_paid", "recently_paid"),
    )

    def explain(self, features: np.ndarray) -> list[dict[str, float]]:
        if features.size == 0:
            return []
        coef = {
            "is_overdue": self.w.overdue,
            "is_due_now": self.w.due_now,
            "is_due_soon": self.w.due_soon,
            "is_recently_paid": self.w.recently_paid,
            "recurrence_confidence": self.w.confidence,
            "schedule_stability": self.w.schedule_stability,
            "amount_is_fixed": self.w.amount_stability,
            "adherence": self.w.adherence,
            "observations_log": self.w.observations_log,
            "missed_cycles": -self.w.missed_cycles,
            "share_of_monthly_spend": -self.w.share_of_monthly_spend,
            "amount_vs_user_median": -self.w.amount_vs_user_median,
            "spend_share": -self.w.spend_share,
            "novelty": -self.w.novelty,
            "success_rate": self.w.success_rate,
            "failed_recent": -self.w.failed_recent,
        }
        days_until = np.clip(features[:, IDX["days_until_due"]], 0.0, 120.0)
        decay = (
            0.9
            * np.exp(-days_until / max(self.w.decay_half_life_days, 1e-6))
            * features[:, IDX["horizon_proximity"]]
        )
        out: list[dict[str, float]] = []
        for row in range(features.shape[0]):
            contrib = dict.fromkeys(coef, 0.0)
            for feat, name in self._TERMS:
                contrib[name] = coef.get(name, 0.0) * float(features[row, IDX[feat]])
            contrib["due_horizon_decay"] = float(decay[row])
            contrib["recurring_periodicity"] = (
                0.6
                * float(features[row, IDX["periodicity_confidence"]])
                * float(features[row, IDX["is_recurring"]])
            )
            out.append(contrib)
        return out

    # ----------------------------------------------------------- persistence
    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.w.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> HeuristicScorer:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(HeuristicWeights.from_dict(data))
