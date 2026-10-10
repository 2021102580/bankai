

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np

from bankai.candidates.generator import CandidateGenerator
from bankai.domain.entities import RecurrenceProfile, Transaction
from bankai.features.builder import (
    build_feature_context,
    build_feature_matrix,
)
from bankai.features.embed import BehaviourEncoder, LatentCollaborative, build_user_profile
from bankai.features.recurrence import detect_recurring
from bankai.logging_setup import get_logger

log = get_logger(__name__)

MIN_HISTORY_DAYS = 45
MAX_POINTS_PER_USER = 40


@dataclass
class LtrExample:
    user_id: str
    as_of: dt.date
    counterparty_id: str
    label: int
    is_hard_negative: bool
    source: str
    base_score: float


@dataclass
class DecisionGroup:
    user_id: str
    as_of: dt.date
    rows: list[LtrExample] = field(default_factory=list)

    @property
    def n_positive(self) -> int:
        return sum(r.label for r in self.rows)


@dataclass
class LtrDataset:
    features: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    group_meta: list[str]
    group_users: list[str]
    n_hard_negatives: int = 0
    n_positives: int = 0
    row_dates: np.ndarray | None = None

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def summary(self) -> dict[str, float | int]:
        return {
            "rows": len(self),
            "groups": int(self.groups.size),
            "positives": self.n_positives,
            "hard_negatives": self.n_hard_negatives,
            "negatives": len(self) - self.n_positives,
            "positives_per_group": float(self.n_positives / max(1, self.groups.size)),
        }


def find_decision_days(
    transactions: list[Transaction],
    *,
    as_of: dt.date,
    horizon: int = 7,
    min_gap_days: int = 3,
) -> list[dt.date]:
    days = sorted(
        {
            t.value_date
            for t in transactions
            if t.direction.value == "outbound"
            and t.status.value == "posted"
            and t.value_date < as_of
        }
    )
    filtered: list[dt.date] = []
    for day in days:
        if (as_of - day).days < MIN_HISTORY_DAYS:
            continue
        if (as_of - day).days > horizon * 60:
            continue
        if filtered and (day - filtered[-1]).days < min_gap_days:
            continue  # collapse clustered debits into a single decision point
        filtered.append(day)
    return filtered[-MAX_POINTS_PER_USER:]


def build_group(
    user_id: str,
    transactions: list[Transaction],
    as_of: dt.date,
    *,
    generator: CandidateGenerator,
    encoder: BehaviourEncoder | None = None,
    latent: LatentCollaborative | None = None,
    label_window_days: int = 3,
    horizon_days: int = 14,
    exploration_negatives: int = 6,
) -> tuple[DecisionGroup, np.ndarray] | None:
    history = [t for t in transactions if t.value_date < as_of]
    if len(history) < 10:
        return None

    profile = build_user_profile(user_id, history, as_of=as_of, inclusive=False)
    recs = detect_recurring(profile.series, as_of=as_of, horizon_days=horizon_days)

    # ---- the pool, exactly as the service would build it ------------------
    generation = generator.generate(profile, recs, as_of=as_of, horizon_days=horizon_days)
    pool: dict[str, str] = {c.counterparty_id: c.source for c in generation.candidates}

    # ---- exploration negatives from the live payee population -------------
    extra_recs: dict[str, RecurrenceProfile] = {}
    available = [cp for cp in _global_population() if cp not in profile.series]
    if available and exploration_negatives > 0:
        rng = np.random.default_rng(
            (abs(hash((user_id, as_of))) + len(generation.candidates)) % (2**32)
        )
        size = min(exploration_negatives, len(available))
        for idx in rng.choice(len(available), size=size, replace=False):
            cp_id = available[int(idx)]
            if cp_id in pool:
                continue
            name, category, _currency = _CP_META.get(cp_id, (cp_id, "uncategorised", "EUR"))
            extra_recs[cp_id] = RecurrenceProfile(
                counterparty_id=cp_id,
                is_recurring=False,
                periodicity_confidence=0.0,
                n_observations=0,
                category=category,
                counterparty_name=name,
            )
            pool[cp_id] = "exploration"

    all_recs: dict[str, RecurrenceProfile] = {**recs, **extra_recs}

    # ---- labels ----------------------------------------------------------
    window_end = as_of + dt.timedelta(days=label_window_days)
    paid_in_window = {
        t.counterparty_id
        for t in transactions
        if t.direction.value == "outbound"
        and t.status.value == "posted"
        and as_of <= t.value_date <= window_end
    }
    positives = {cp for cp in pool if cp in paid_in_window}
    if not positives:
        return None

    group = DecisionGroup(user_id=user_id, as_of=as_of)
    for cp_id, source in pool.items():
        label = 1 if cp_id in positives else 0
        group.rows.append(
            LtrExample(
                user_id=user_id,
                as_of=as_of,
                counterparty_id=cp_id,
                label=label,
                is_hard_negative=(label == 0 and source != "exploration"),
                source=source,
                base_score=1.0 if label else 0.0,
            )
        )

    # ---- features, via the shared training/serving context builder -------
    ids = [r.counterparty_id for r in group.rows]
    ctx = build_feature_context(
        profile,
        all_recs,
        as_of=as_of,
        horizon_days=horizon_days,
        encoder=encoder,
        latent=latent,
        counterparty_ids=list(all_recs),
    )
    return group, build_feature_matrix(ids, ctx)


# --------------------------------------------------------------------------- #
# Global payee population (exploration negatives)
# --------------------------------------------------------------------------- #
_CP_META: dict[str, tuple[str, str, str]] = {}


def register_counterparty_population(
    rows: list[tuple[str, str, str, str]],
) -> None:
    _CP_META.clear()
    for cp_id, name, category, currency in rows:
        _CP_META[cp_id] = (name, category, currency)


def _global_population() -> list[str]:
    return list(_CP_META.keys())


def build_population_from_transactions(
    transactions: list[Transaction], min_users: int = 3
) -> list[tuple[str, str, str, str]]:
    seen: dict[str, tuple[set[str], str, str, str]] = {}
    for t in transactions:
        users, name, category, currency = seen.get(
            t.counterparty_id, (set(), t.counterparty_name, t.counterparty_category, t.currency)
        )
        users.add(t.user_id)
        seen[t.counterparty_id] = (users, name, category, currency)
    return [
        (cp_id, name, category, currency)
        for cp_id, (users, name, category, currency) in seen.items()
        if len(users) >= min_users
    ]


__all__ = [
    "DecisionGroup",
    "LtrDataset",
    "LtrExample",
    "build_group",
    "build_population_from_transactions",
    "find_decision_days",
    "register_counterparty_population",
]
