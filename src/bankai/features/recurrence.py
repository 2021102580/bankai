

from __future__ import annotations

import datetime as dt
import itertools
from dataclasses import dataclass

import numpy as np

from bankai.domain.entities import RecurrenceProfile, RecurrenceStatus
from bankai.features.calendar import (
    CANONICAL_PERIODS,
    MONTH_MEAN_DAYS,
    add_months,
    bounded,
    circular_concentration,
    circular_std,
    day_of_month_angles,
    robust_cv,
)
from bankai.features.series import CounterpartySeries


MIN_OBSERVATIONS = 3
PERIOD_TOLERANCE_REL = 0.10
PERIOD_TOLERANCE_ABS = 4.0

CALENDAR_FAMILIES: tuple[tuple[float, float, int], ...] = (
    (26.0, 32.0, 1),  # monthly
    (84.0, 100.0, 3),  # quarterly
    (178.0, 190.0, 6),  # semi-annual
    (360.0, 372.0, 12),  # annual
)

STALE_MULTIPLIER = 3.0
DUE_WINDOW_DAYS = 3
RECENT_PAYMENT_DAYS = 7
ABANDONED_MULTIPLIER = 2.5
PARSIMONY_EPSILON = 0.03
MAX_CYCLE_MULTIPLE = 6


def _tolerance(period: float) -> float:
    return max(PERIOD_TOLERANCE_ABS, PERIOD_TOLERANCE_REL * period)


def _calendar_family(period: float) -> int | None:
    for low, high, months in CALENDAR_FAMILIES:
        if low <= period <= high:
            return months
    return None


def _predict_next(prev: dt.date, period: float, cycles: int) -> dt.date:
    family = _calendar_family(period)
    if family is not None:
        return add_months(prev, family * cycles)
    return prev + dt.timedelta(days=round(period * cycles))


def _cycles_between(prev: dt.date, nxt: dt.date, period: float) -> int:
    family = _calendar_family(period)
    if family is not None:
        months = (nxt.year - prev.year) * 12 + (nxt.month - prev.month)
        raw = months / family
    else:
        raw = (nxt - prev).days / period
    return min(max(round(raw), 1), MAX_CYCLE_MULTIPLE)


@dataclass(frozen=True, slots=True)
class PeriodFit:
    period: float
    score: float
    matched_gaps: int
    coverage: float
    absolute_error: float
    mean_skipped_cycles: float = 0.0


def fit_period(dates: list[dt.date]) -> PeriodFit:
    clean = sorted(set(dates))
    if len(clean) < 2:
        return PeriodFit(
            period=MONTH_MEAN_DAYS,
            score=0.0,
            matched_gaps=0,
            coverage=0.0,
            absolute_error=float("inf"),
        )

    prior = dict.fromkeys(CANONICAL_PERIODS, 1.0)
    for p in (7.0, 14.0, 28.0, 30.0, 30.5, 31.0, 91.0, 182.0, 365.0):
        prior[p] = 1.08  # slight preference for real-world cadences
    for p in (1.0, 2.0, 3.0, 5.0, 6.0, 26.0, 35.0, 36.0, 75.0, 78.0, 150.0, 156.0, 273.0):
        prior[p] = 0.95

    fits: list[PeriodFit] = []
    for period in CANONICAL_PERIODS:
        tol = _tolerance(period)
        errors: list[float] = []
        skips: list[float] = []
        for prev, nxt in itertools.pairwise(clean):
            k = _cycles_between(prev, nxt, period)
            predicted = _predict_next(prev, period, k)
            errors.append(abs((nxt - predicted).days))
            skips.append(k - 1.0)
            if len(errors) >= 240:  # bound the cost on pathological histories
                break
        err = np.asarray(errors, dtype=float)
        coverage = float((err <= tol).mean())
        if coverage < 0.60:
            continue
        mean_err = float(err.mean())
        mean_skips = float(np.mean(skips))
        scale = max(1.0, 0.05 * period)
        residual_quality = float(np.exp(-mean_err / scale))
        score = (
            0.45 * residual_quality
            + 0.30 * coverage
            + 0.15 * prior.get(period, 1.0)
            + 0.10 * (1.0 - min(mean_skips, 3.0) / 3.0)
        )
        fits.append(
            PeriodFit(
                period=float(period),
                score=float(score),
                matched_gaps=int((err <= tol).sum()),
                coverage=coverage,
                absolute_error=mean_err,
                mean_skipped_cycles=mean_skips,
            )
        )

    if not fits:
        return PeriodFit(
            period=MONTH_MEAN_DAYS,
            score=0.0,
            matched_gaps=0,
            coverage=0.0,
            absolute_error=float("inf"),
        )

    best_score = max(f.score for f in fits)
    finalists = [f for f in fits if f.score >= best_score - PARSIMONY_EPSILON]
    longest = max(f.period for f in finalists)
    return max((f for f in finalists if f.period == longest), key=lambda f: f.score)


def _schedule_stability(series: CounterpartySeries, period: float) -> float:
    gaps = series.days_between_points()
    if gaps.size == 0:
        return 0.0
    mean_gap = float(np.mean(gaps))
    if mean_gap <= 0:
        return 0.0
    gap_cv = robust_cv(gaps, floor=1.0)
    gap_component = 1.0 / (1.0 + gap_cv)

    if period <= 45.0:
        angles = day_of_month_angles(
            [d.toordinal() % 31 for d in series.dates], period=MONTH_MEAN_DAYS
        )
        dom_std = circular_std(angles, MONTH_MEAN_DAYS)
        dom_component = float(np.exp(-dom_std / 4.0))
        weekday = series.typical_weekday()
        wd_component = 0.0
        if weekday is not None:
            wd_hits = sum(1 for d in series.dates if d.weekday() == weekday) / series.n
            wd_component = wd_hits
        return bounded(0.45 * gap_component + 0.30 * dom_component + 0.25 * wd_component)

    angles = day_of_month_angles([d.toordinal() for d in series.dates], period=max(period, 1.0))
    phase_component = circular_concentration(angles)
    return bounded(0.65 * gap_component + 0.35 * phase_component)


def _project_next(series: CounterpartySeries, period: float, as_of: dt.date) -> tuple[dt.date, int]:
    last = series.last_date
    if last is None:
        return as_of, 0
    missed = _missed_cycles(series, period, as_of)
    nxt = last + dt.timedelta(days=round(period))
    if missed >= 4:
        return as_of, missed
    return nxt, missed


def _missed_cycles(series: CounterpartySeries, period: float, as_of: dt.date) -> int:
    last = series.last_date
    if last is None or period <= 0:
        return 0
    elapsed = (as_of - last).days
    return max(0, round(elapsed / period) - 1)


def _cycle_bounds(anchor: dt.date, period: int) -> tuple[dt.date, dt.date]:
    half = max(period // 2, 3)
    return anchor - dt.timedelta(days=half), anchor + dt.timedelta(days=half)


def classify_status(
    series: CounterpartySeries,
    *,
    as_of: dt.date,
    next_due: dt.date | None,
    period: float | None,
    horizon_days: int,
    missed_cycles: int = 0,
) -> RecurrenceStatus:
    if next_due is None or not period or period <= 0:
        return RecurrenceStatus.ONE_OFF

    days_until = (next_due - as_of).days
    last = series.last_date

    if missed_cycles >= 2:
        return RecurrenceStatus.RECURRING
    if days_until < -DUE_WINDOW_DAYS:
        return RecurrenceStatus.OVERDUE
    if days_until <= DUE_WINDOW_DAYS:
        return RecurrenceStatus.DUE_NOW
    days_since = (as_of - last).days if last is not None else 999
    if days_since <= RECENT_PAYMENT_DAYS:
        return RecurrenceStatus.RECENTLY_PAID
    if days_until <= horizon_days:
        return RecurrenceStatus.DUE_SOON
    return RecurrenceStatus.RECURRING


def analyse_series(
    series: CounterpartySeries,
    *,
    as_of: dt.date,
    horizon_days: int = 14,
    min_amount: float = 0.5,
) -> RecurrenceProfile:
    as_of_ord = as_of.toordinal()
    points = [p for p in series.points if p.value_date.toordinal() <= as_of_ord]
    n = len(points)
    amounts = np.asarray([p.amount for p in points], dtype=float)

    if n == 0:
        return RecurrenceProfile(
            counterparty_id=series.counterparty_id,
            is_recurring=False,
            periodicity_confidence=0.0,
            n_observations=0,
            category=series.category,
            counterparty_name=series.counterparty_name,
        )

    if n == 1:
        return RecurrenceProfile(
            counterparty_id=series.counterparty_id,
            is_recurring=False,
            periodicity_confidence=0.0,
            n_observations=1,
            adherence=1.0,
            last_paid_on=points[-1].value_date,
            median_amount=float(amounts[0]),
            expected_amount=float(amounts[0]),
            amount_low=float(amounts[0]),
            amount_high=float(amounts[0]),
            category=series.category,
            counterparty_name=series.counterparty_name,
            status=RecurrenceStatus.ONE_OFF,
            typical_weekday=points[-1].value_date.weekday(),
        )

    ordinals = np.asarray([p.value_date.toordinal() for p in points], dtype=float)
    gaps = np.diff(ordinals)
    clean_dates = sorted({p.value_date for p in points})
    fit = fit_period(clean_dates)

    period = fit.period if fit.coverage > 0 else None
    periodicity_confidence = bounded(fit.coverage * (0.6 + 0.4 * fit.score))

    is_recurring = bool(
        n >= MIN_OBSERVATIONS
        and period is not None
        and periodicity_confidence >= 0.45
        and series.days_since_last(as_of) is not None
        and series.days_since_last(as_of) <= (period or 0) * 2.2
    )

    first = points[0].value_date

    # -- amounts -------------------------------------------------------------
    amount_cv = robust_cv(amounts[-8:])
    median_amount = float(np.median(amounts[-8:]))
    abs_dispersion = amount_cv * max(median_amount, 0.0)
    amount_stability = float(
        np.clip(1.0 - abs_dispersion / max(2.0, 0.10 * max(median_amount, 1.0)), 0.0, 1.0)
    )
    amount_stability = float(
        0.65 * amount_stability + 0.35 * bounded(1.0 / (1.0 + 6.0 * amount_cv))
    )

    last = points[-1].value_date
    next_due: dt.date | None = None
    missed = 0
    if period is not None:
        next_due, missed = _project_next(series, period, as_of)

    schedule_stability = _schedule_stability(series, period) if (period and is_recurring) else 0.0
    dedup_gaps = np.diff([d.toordinal() for d in clean_dates])
    gap_cv = robust_cv(dedup_gaps if dedup_gaps.size else gaps, floor=1.0)
    regularity = float(1.0 / (1.0 + gap_cv)) if (dedup_gaps.size or gaps.size) else 0.0

    adherence = 0.0
    if period and period > 0 and first is not None:
        span = max(1.0, float((as_of - first).days))
        expected = max(1.0, span / period)
        adherence = float(np.clip(n / expected, 0.0, 1.0))

    status = classify_status(
        series,
        as_of=as_of,
        next_due=next_due,
        period=period,
        horizon_days=horizon_days,
        missed_cycles=missed,
    )
    expected_amount = series.expected_amount(period, as_of=as_of)
    if not is_recurring and period is None:
        status = RecurrenceStatus.ONE_OFF
    low, high = series.amount_quantiles()
    spread = max(0.01, 0.02 * expected_amount)

    return RecurrenceProfile(
        counterparty_id=series.counterparty_id,
        is_recurring=is_recurring,
        period_days=period,
        periodicity_confidence=float(periodicity_confidence),
        amount_stability=float(np.clip(amount_stability, 0.0, 1.0)),
        schedule_stability=float(np.clip(schedule_stability, 0.0, 1.0)),
        regularity=float(np.clip(regularity, 0.0, 1.0)),
        n_observations=n,
        adherence=adherence,
        last_paid_on=last,
        next_due_on=next_due,
        days_until_due=(next_due - as_of).days if next_due else None,
        missed_cycles=missed,
        median_amount=median_amount,
        amount_dispersion=float(amount_cv),
        expected_amount=float(expected_amount),
        amount_low=float(max(low, expected_amount - spread)),
        amount_high=float(max(high, expected_amount + spread)),
        cycle_index=missed + 1,
        status=status,
        typical_weekday=last.weekday(),
        category=series.category,
        counterparty_name=series.counterparty_name,
    )


def detect_recurring(
    series_by_cp: dict[str, CounterpartySeries],
    *,
    as_of: dt.date,
    horizon_days: int = 14,
) -> dict[str, RecurrenceProfile]:
    return {
        cp_id: analyse_series(series, as_of=as_of, horizon_days=horizon_days)
        for cp_id, series in series_by_cp.items()
    }


def discover_cycles(
    profiles: dict[str, RecurrenceProfile], *, min_count: int = 3, min_phase_overlap: float = 0.55
) -> list[list[str]]:
    buckets: dict[int, list[str]] = {}
    for cp_id, prof in profiles.items():
        if not prof.is_recurring or prof.next_due_on is None or not prof.period_days:
            continue
        if prof.days_until_due is None or prof.days_until_due < -5 or prof.days_until_due > 20:
            continue
        key = round(prof.days_until_due / 3.0)
        buckets.setdefault(key, []).append(cp_id)

    bundles: list[list[str]] = []
    for members in buckets.values():
        if len(members) < min_count:
            continue
        same_window = all(
            profiles[m].next_due_on is not None
            and abs((profiles[m].next_due_on - profiles[members[0]].next_due_on).days) <= 4
            for m in members
        )
        if same_window and len(members) >= min_count:
            bundles.append(sorted(members))
    return bundles


__all__ = [
    "PeriodFit",
    "analyse_series",
    "classify_status",
    "detect_recurring",
    "discover_cycles",
    "fit_period",
]
