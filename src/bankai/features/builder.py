
from __future__ import annotations

import datetime as dt
import math
from collections.abc import Sequence

import numpy as np

from bankai.domain.entities import RecurrenceProfile, RecurrenceStatus, Transaction
from bankai.features.calendar import (
    days_until_month_end,
    is_last_week_of_month,
    quarter_bucket,
    weekend_ordinal,
)
from bankai.features.embed import BehaviourEncoder, UserProfile
from bankai.features.series import CounterpartySeries

FEATURE_NAMES: tuple[str, ...] = (
    # --- recurrence evidence -------------------------------------------------
    "is_recurring",
    "periodicity_confidence",
    "schedule_stability",
    "amount_stability",
    "recurrence_regularity",
    "recurrence_confidence",
    "period_days",
    "period_log",
    "period_is_monthly",
    "period_is_weekly",
    "period_is_quarterly",
    "n_observations",
    "observations_log",
    "adherence",
    "span_days",
    "distinct_months",
    "active_month_ratio",
    "missed_cycles",
    # --- due-date proximity --------------------------------------------------
    "days_until_due",
    "days_until_due_abs",
    "is_due_now",
    "is_due_soon",
    "is_overdue",
    "is_recently_paid",
    "days_since_last",
    "recency_ratio",
    "recency_rank_pct",
    # --- amount --------------------------------------------------------------
    "expected_amount",
    "expected_amount_log",
    "amount_vs_user_median",
    "amount_log",
    "amount_dispersion",
    "amount_is_fixed",
    "amount_matches_last",
    "share_of_monthly_spend",
    # --- affinity / habit ---------------------------------------------------
    "tx_share",
    "spend_share",
    "spend_90d",
    "spend_90d_log",
    "count_90d",
    "count_365d",
    "category_share",
    "category_affinity",
    "is_dominant_category",
    "success_rate",
    "failed_recent",
    "channel_diversity",
    # --- similarity ----------------------------------------------------------
    "text_similarity",
    "latent_similarity",
    "novelty",
    "is_known_payee",
    # --- calendar context ---------------------------------------------------
    "as_of_weekday",
    "as_of_is_weekend",
    "as_of_days_to_month_end",
    "as_of_last_week_of_month",
    "as_of_quarter_boundary",
    "due_weekday",
    "due_weekday_match",
    "due_is_month_boundary",
    "due_is_mid_month",
    # --- portfolio / business ----------------------------------------------
    "n_user_counterparties",
    "user_recurring_count",
    "user_tx_count_log",
    "is_new_this_month",
    "days_since_first_log",
    "horizon_proximity",
    # --- position in the billing cycle ------------------------------------
    # Raw days-to-due is not comparable across a weekly coffee round and a
    # quarterly insurance premium; its position within the cycle is.
    "due_position",
    "due_position_abs",
    "is_at_cycle_start",
    "is_mid_cycle",
    # --- payday alignment --------------------------------------------------
    # A large share of customers sweep their bills within a few days of salary.
    "payday_offset_days",
    "is_near_payday",
    "payday_exists",
    # --- batch / co-payment behaviour -------------------------------------
    "copay_rate",
    "n_paid_last_active_day",
    "days_since_last_active_day",
    "user_payments_per_active_day",
    "is_batching_user",
    # --- amount dynamics ----------------------------------------------------
    "amount_delta_pct",
    "amount_is_seasonal",
    # --- calendar shape of the last payment ---------------------------------
    "last_day_of_month",
    "last_weekday",
    "last_month_sin",
    "last_month_cos",
)

FEATURE_VERSION = "4"
N_FEATURES = len(FEATURE_NAMES)

_HORIZON_PROXIMITY = {
    RecurrenceStatus.OVERDUE: 1.00,
    RecurrenceStatus.DUE_NOW: 0.95,
    RecurrenceStatus.DUE_SOON: 0.80,
    RecurrenceStatus.RECENTLY_PAID: 0.35,
    RecurrenceStatus.RECURRING: 0.20,
    RecurrenceStatus.ONE_OFF: 0.05,
    RecurrenceStatus.UNKNOWN: 0.05,
}


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return float(a / b) if abs(b) > 1e-9 else default


def _safe_log(x: float) -> float:
    return float(math.log1p(max(0.0, x)))


def _clamp(x: float, lo: float = -20.0, hi: float = 20.0) -> float:
    if not np.isfinite(x):
        return 0.0
    return float(min(max(x, lo), hi))


class FeatureContext:
    def __init__(
        self,
        profile: UserProfile,
        profiles: dict[str, RecurrenceProfile],
        *,
        as_of: dt.date,
        horizon_days: int,
        encoder: BehaviourEncoder | None = None,
        latent_scores: dict[str, float] | None = None,
        centroid: np.ndarray | None = None,
    ) -> None:
        self.profile = profile
        self.profiles = profiles
        self.as_of = as_of
        self.horizon_days = horizon_days
        self.encoder = encoder
        self.latent_scores = latent_scores or {}
        self.centroid = centroid

        self.series: dict[str, CounterpartySeries] = profile.series
        n = max(1, sum(s.n for s in self.series.values()))
        spend = max(1e-6, profile.outbound_total)

        self.user_median_amount = profile.median_tx_amount
        self.tx_share: dict[str, float] = {cp: s.n / n for cp, s in self.series.items()}
        self.spend_share: dict[str, float] = {
            cp: s.total_amount() / spend for cp, s in self.series.items()
        }
        self.spend_90d: dict[str, float] = {
            cp: s.amount_in_window(as_of, 90) for cp, s in self.series.items()
        }
        self.count_90d: dict[str, float] = {
            cp: float(s.count_in_window(as_of, 90)) for cp, s in self.series.items()
        }
        self.count_365d: dict[str, float] = {
            cp: float(s.count_in_window(as_of, 365)) for cp, s in self.series.items()
        }
        self.monthly_spend: float = max(1.0, profile.mean_monthly_outbound)
        self.n_counterparties = max(1, profile.n_counterparties)
        self.user_tx_count = max(1, profile.n_transactions)
        self.user_recurring_count = max(1, sum(1 for p in profiles.values() if p.is_recurring))

        recencies = [s.days_since_last(as_of) or 9999 for s in self.series.values()]
        self._recency_values = recencies

        self.category_concentration: dict[str, float] = {}
        for cat, amount in profile.category_spend.items():
            self.category_concentration[cat] = _safe_div(
                amount * amount, sum(v * v for v in profile.category_spend.values())
            )

        # --- batch-payment aggregates, shared across all candidates ---------
        self.batch_days = profile.batch_day_ordinals
        self.salary_day = profile.salary_day_of_month
        self.n_paid_last_active = profile.n_paid_on_last_active_day
        self.days_since_last_active = profile.days_since_last_active_day or 999
        self.payments_per_active_day = profile.payments_per_active_day
        self.is_batching_user = 1.0 if profile.payments_per_active_day >= 1.5 else 0.0

    # ------------------------------------------------------------------ helpers
    def copay_rate(self, series: CounterpartySeries | None) -> float:
        if series is None or series.n == 0 or not self.batch_days:
            return 0.0
        hits = sum(1 for p in series.points if p.value_date.toordinal() in self.batch_days)
        return _clamp(hits / series.n, 0, 1)

    def recency_rank_pct(self, days_since: int | None) -> float:
        if days_since is None or not self._recency_values:
            return 0.5
        return float(
            sum(1 for r in self._recency_values if r <= days_since) / len(self._recency_values)
        )

    def text_similarity(self, counterparty_id: str) -> float:
        if self.encoder is None or self.centroid is None or not self.encoder.fitted:
            return 0.0
        vec = self.encoder.item_vector(counterparty_id)
        if vec is None:
            return 0.0
        return _clamp(float(np.dot(vec, self.centroid)), -1.0, 1.0)

    def latent_similarity(self, counterparty_id: str) -> float:
        return _clamp(self.latent_scores.get(counterparty_id, 0.0), -1.0, 1.0)


def candidate_features(
    counterparty_id: str,
    ctx: FeatureContext,
) -> dict[str, float]:
    profile: UserProfile = ctx.profile
    as_of: dt.date = ctx.as_of
    rec: RecurrenceProfile = ctx.profiles.get(
        counterparty_id,
        RecurrenceProfile(
            counterparty_id=counterparty_id,
            is_recurring=False,
            periodicity_confidence=0.0,
            n_observations=0,
        ),
    )
    series: CounterpartySeries | None = ctx.series.get(counterparty_id)

    expected_amount = rec.expected_amount or (series.median_amount() if series else 0.0) or 0.0
    days_since = series.days_since_last(as_of) if series else None
    days_until = rec.days_until_due
    period = rec.period_days or 0.0
    first = series.first_date if series else None
    span_days = float((as_of - first).days) if first else 0.0
    last = series.last_date if series else None

    # --- position in the cycle -------------------------------------------
    due_position = _clamp((days_until or 0.0) / period, -3, 3) if period > 0 else 0.0
    due_position_abs = abs(due_position)

    # --- payday alignment --------------------------------------------------
    payday_offset = 0.0
    if ctx.salary_day and last is not None:
        typical_day = rec.typical_weekday is not None and (
            series.billing_day_histogram() if series else {}
        )
        dom_day = max(typical_day.items(), key=lambda kv: kv[1])[0] if typical_day else last.day
        raw = abs(dom_day - ctx.salary_day)
        payday_offset = float(min(raw, 31 - raw))  # wrap around the month
    # --- amount dynamics ---------------------------------------------------
    last_amount = series.last_amount if series else None
    amount_delta = (
        _clamp((expected_amount - last_amount) / max(abs(last_amount), 1.0), -1, 1)
        if last_amount
        else 0.0
    )
    amount_is_seasonal = 0.0
    if series is not None and series.n >= 8:
        monthly = series.monthly_amount_series()
        by_month: dict[int, list[float]] = {}
        for (_y, m), amount in monthly.items():
            by_month.setdefault(m, []).append(amount)
        spreads = [
            (max(v) - min(v)) / max(np.mean(v), 1.0) for v in by_month.values() if len(v) >= 2
        ]
        if spreads:
            amount_is_seasonal = _clamp(float(np.mean(spreads)) / 0.4, 0, 1)
    status = rec.status
    feat: dict[str, float] = {
        "is_recurring": 1.0 if rec.is_recurring else 0.0,
        "periodicity_confidence": rec.periodicity_confidence,
        "schedule_stability": rec.schedule_stability,
        "amount_stability": rec.amount_stability,
        "recurrence_regularity": rec.regularity,
        "recurrence_confidence": rec.confidence,
        "period_days": _clamp(period, 0, 800),
        "period_log": _safe_log(period),
        "period_is_monthly": 1.0 if 25.0 <= period <= 33.0 else 0.0,
        "period_is_weekly": 1.0 if 5.0 <= period <= 9.0 else 0.0,
        "period_is_quarterly": 1.0 if 80.0 <= period <= 100.0 else 0.0,
        "n_observations": float(rec.n_observations),
        "observations_log": _safe_log(float(rec.n_observations)),
        "adherence": rec.adherence,
        "span_days": _clamp(span_days, 0, 4000),
        "distinct_months": float(series.distinct_months()) if series else 0.0,
        "active_month_ratio": series.active_month_ratio(as_of) if series else 0.0,
        "missed_cycles": float(min(rec.missed_cycles, 24)),
        # due-date proximity
        "days_until_due": _clamp(float(days_until) if days_until is not None else 999.0, -400, 400),
        "days_until_due_abs": _clamp(
            abs(float(days_until)) if days_until is not None else 999.0, 0, 400
        ),
        "is_due_now": 1.0 if status is RecurrenceStatus.DUE_NOW else 0.0,
        "is_due_soon": 1.0 if status is RecurrenceStatus.DUE_SOON else 0.0,
        "is_overdue": 1.0 if status is RecurrenceStatus.OVERDUE else 0.0,
        "is_recently_paid": 1.0 if status is RecurrenceStatus.RECENTLY_PAID else 0.0,
        "days_since_last": _clamp(float(days_since) if days_since is not None else 999.0, 0, 2000),
        "recency_ratio": _clamp(_safe_div(float(days_since or 0), max(period, 1.0)), 0, 20),
        "recency_rank_pct": ctx.recency_rank_pct(days_since),
        # amount
        "expected_amount": _clamp(expected_amount, 0, 1e6),
        "expected_amount_log": _safe_log(expected_amount),
        "amount_vs_user_median": _clamp(
            _safe_div(expected_amount, max(ctx.user_median_amount, 1.0)), 0, 500
        ),
        "amount_log": _safe_log(expected_amount),
        "amount_dispersion": _clamp(rec.amount_dispersion, 0, 20),
        "amount_is_fixed": rec.amount_stability,
        "amount_matches_last": (
            1.0
            if series
            and series.last_amount is not None
            and abs(series.last_amount - expected_amount) <= max(0.02 * expected_amount, 0.5)
            else 0.0
        ),
        "share_of_monthly_spend": _clamp(_safe_div(expected_amount, ctx.monthly_spend), 0, 5),
        # affinity
        "tx_share": _clamp(ctx.tx_share.get(counterparty_id, 0.0), 0, 1),
        "spend_share": _clamp(ctx.spend_share.get(counterparty_id, 0.0), 0, 1),
        "spend_90d": _clamp(ctx.spend_90d.get(counterparty_id, 0.0), 0, 1e6),
        "spend_90d_log": _safe_log(ctx.spend_90d.get(counterparty_id, 0.0)),
        "count_90d": ctx.count_90d.get(counterparty_id, 0.0),
        "count_365d": ctx.count_365d.get(counterparty_id, 0.0),
        "category_share": _clamp(profile.category_share(rec.category), 0, 1),
        "category_affinity": _clamp(ctx.category_concentration.get(rec.category, 0.0), 0, 1),
        "is_dominant_category": (
            1.0
            if rec.category
            == max(profile.category_spend, key=profile.category_spend.get, default="")
            else 0.0
        ),
        "success_rate": _clamp(1.0 - _safe_div(series.failed_count, max(series.n, 1)), 0, 1)
        if series
        else 0.5,
        "failed_recent": float(min(profile.failed_recent, 10)),
        "channel_diversity": (
            len({p.channel for p in series.points}) if series and series.n else 1.0
        ),
        # similarity
        "text_similarity": ctx.text_similarity(counterparty_id),
        "latent_similarity": ctx.latent_similarity(counterparty_id),
        "novelty": 1.0 - ctx.text_similarity(counterparty_id),
        "is_known_payee": 1.0 if series is not None else 0.0,
        # calendar
        "as_of_weekday": float(as_of.weekday()),
        "as_of_is_weekend": weekend_ordinal(as_of),
        "as_of_days_to_month_end": float(days_until_month_end(as_of)),
        "as_of_last_week_of_month": 1.0 if is_last_week_of_month(as_of) else 0.0,
        "as_of_quarter_boundary": 1.0 if as_of.day <= 3 or as_of.day >= 27 else 0.0,
        "due_weekday": float(rec.next_due_on.weekday()) if rec.next_due_on else 4.0,
        "due_weekday_match": (
            1.0 if rec.next_due_on and rec.next_due_on.weekday() == as_of.weekday() else 0.0
        ),
        "due_is_month_boundary": (
            1.0
            if rec.next_due_on and (rec.next_due_on.day <= 3 or rec.next_due_on.day >= 28)
            else 0.0
        ),
        "due_is_mid_month": (1.0 if rec.next_due_on and 8 <= rec.next_due_on.day <= 22 else 0.0),
        # portfolio
        "n_user_counterparties": float(ctx.n_counterparties),
        "user_recurring_count": float(ctx.user_recurring_count),
        "user_tx_count_log": _safe_log(float(ctx.user_tx_count)),
        "is_new_this_month": (1.0 if first and (as_of - first).days <= 31 else 0.0),
        "days_since_first_log": _safe_log(span_days),
        "horizon_proximity": _HORIZON_PROXIMITY.get(status, 0.05),
        # cycle position
        "due_position": due_position,
        "due_position_abs": due_position_abs,
        "is_at_cycle_start": 1.0 if -0.15 <= due_position <= 0.15 else 0.0,
        "is_mid_cycle": 1.0 if 0.35 <= due_position <= 0.75 else 0.0,
        # payday alignment
        "payday_offset_days": payday_offset,
        "is_near_payday": 1.0 if 0 < payday_offset <= 4.0 else 0.0,
        "payday_exists": 1.0 if ctx.salary_day else 0.0,
        # batch / co-payment
        "copay_rate": ctx.copay_rate(series),
        "n_paid_last_active_day": float(ctx.n_paid_last_active),
        "days_since_last_active_day": float(ctx.days_since_last_active),
        "user_payments_per_active_day": _clamp(ctx.payments_per_active_day, 0, 8),
        "is_batching_user": ctx.is_batching_user,
        # amount dynamics
        "amount_delta_pct": amount_delta,
        "amount_is_seasonal": amount_is_seasonal,
        # calendar shape of the last payment
        "last_day_of_month": float(last.day) if last else 0.0,
        "last_weekday": float(last.weekday()) if last else 4.0,
        "last_month_sin": float(np.sin(2.0 * np.pi * ((last.month - 1) / 12.0))) if last else 0.0,
        "last_month_cos": float(np.cos(2.0 * np.pi * ((last.month - 1) / 12.0))) if last else 0.0,
    }
    return feat


def build_feature_matrix(counterparty_ids: Sequence[str], ctx: FeatureContext) -> np.ndarray:
    rows: list[list[float]] = []
    for cp_id in counterparty_ids:
        f = candidate_features(cp_id, ctx)
        missing = set(FEATURE_NAMES) - set(f)
        if missing:  # pragma: no cover - guards contract drift
            raise KeyError(f"feature contract drift, missing: {sorted(missing)}")
        rows.append([_clamp(f[name]) for name in FEATURE_NAMES])
    if not rows:
        return np.zeros((0, N_FEATURES), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def feature_importance_frame(feature_names: Sequence[str], importances: Sequence[float]):
    import pandas as pd

    frame = pd.DataFrame({"feature": list(feature_names), "gain": list(importances)})
    return frame.sort_values("gain", ascending=False).reset_index(drop=True)


def transactions_upto(transactions: Sequence[Transaction], as_of: dt.date) -> list[Transaction]:
    cutoff = as_of.toordinal()
    return [t for t in transactions if t.value_date.toordinal() <= cutoff]


def build_feature_context(
    profile: UserProfile,
    profiles: dict[str, RecurrenceProfile],
    *,
    as_of: dt.date,
    horizon_days: int,
    encoder: BehaviourEncoder | None = None,
    latent: object | None = None,
    counterparty_ids: Sequence[str] | None = None,
) -> FeatureContext:
    latent_scores: dict[str, float] | None = None
    if latent is not None and getattr(latent, "fitted", False):
        vector = latent.user_vector(profile.user_id)  # type: ignore[attr-defined]
        if vector is not None:
            ids = (
                list(counterparty_ids)
                if counterparty_ids is not None
                else (list(profile.series) + list(profiles))
            )
            latent_scores = {cp: float(latent.score(vector, cp)) for cp in ids}  # type: ignore[attr-defined]
    centroid = None
    if encoder is not None and getattr(encoder, "fitted", False):
        centroid = encoder.user_centroid(profile.series, as_of=as_of)
    return FeatureContext(
        profile,
        profiles,
        as_of=as_of,
        horizon_days=horizon_days,
        encoder=encoder,
        latent_scores=latent_scores,
        centroid=centroid,
    )


__all__ = [
    "FEATURE_NAMES",
    "FEATURE_VERSION",
    "N_FEATURES",
    "FeatureContext",
    "build_feature_context",
    "build_feature_matrix",
    "candidate_features",
    "feature_importance_frame",
    "quarter_bucket",
    "transactions_upto",
]
