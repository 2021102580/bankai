

from __future__ import annotations

import calendar
import datetime as dt
from functools import lru_cache

import numpy as np

MONTH_MEAN_DAYS = 30.436875
WEEK = 7.0
YEAR = 365.2425

CANONICAL_PERIODS: tuple[float, ...] = (
    1.0,
    2.0,
    3.0,
    5.0,
    6.0,
    7.0,
    10.0,
    12.0,
    14.0,
    15.0,
    20.0,
    21.0,
    24.0,
    26.0,
    28.0,
    30.0,
    30.5,
    31.0,
    35.0,
    36.0,
    40.0,
    42.0,
    45.0,
    50.0,
    52.0,
    56.0,
    60.0,
    61.0,
    70.0,
    72.0,
    75.0,
    78.0,
    84.0,
    90.0,
    91.0,
    100.0,
    105.0,
    120.0,
    122.0,
    150.0,
    156.0,
    180.0,
    182.0,
    182.5,
    270.0,
    273.0,
    365.0,
    365.25,
)


def day_of_month_angles(dates: np.ndarray, period: float = MONTH_MEAN_DAYS) -> np.ndarray:
    return 2.0 * np.pi * np.asarray(dates, dtype=float) / period


def circular_mean(angles: np.ndarray) -> float:
    if angles.size == 0:
        return float("nan")
    return float(np.arctan2(np.sin(angles).mean(), np.cos(angles).mean()))


def circular_std(angles: np.ndarray, period: float = MONTH_MEAN_DAYS) -> float:
    if angles.size == 0:
        return float("nan")
    if angles.size == 1:
        return 0.0
    c = float(np.cos(angles).mean())
    s = float(np.sin(angles).mean())
    resultant = float(np.hypot(c, s))
    resultant = min(max(resultant, 1e-9), 1.0)
    return float(np.sqrt(-2.0 * np.log(resultant)) * period / (2.0 * np.pi))


def circular_concentration(angles: np.ndarray) -> float:
    if angles.size == 0:
        return 0.0
    return float(np.clip(np.hypot(np.cos(angles).mean(), np.sin(angles).mean()), 0.0, 1.0))


def robust_cv(values: np.ndarray, floor: float = 1e-6) -> float:
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return 0.0
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return float(1.4826 * mad / max(abs(median), floor))


def bounded(value: float, low: float = 0.0, high: float = 1.0) -> float:
    if not np.isfinite(value):
        return low
    return float(min(max(value, low), high))


def stability_from_dispersion(dispersion: float, tolerance: float) -> float:
    if not np.isfinite(dispersion):
        return 0.0
    if tolerance <= 0:
        return 0.0
    return bounded(1.0 - dispersion / tolerance)


def days_in_month(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


def month_end(value: dt.date) -> dt.date:
    return dt.date(value.year, value.month, days_in_month(value.year, value.month))


def month_start(value: dt.date) -> dt.date:
    return value.replace(day=1)


def add_months(value: dt.date, months: int) -> dt.date:
    total = value.month - 1 + months
    year = value.year + total // 12
    month = total % 12 + 1
    return dt.date(year, month, min(value.day, days_in_month(year, month)))


def month_key(value: dt.date) -> tuple[int, int]:
    return (value.year, value.month)


def days_until_month_end(value: dt.date) -> int:
    return (month_end(value) - value).days


def is_last_week_of_month(value: dt.date) -> bool:
    return days_until_month_end(value) <= 6


def normalised_weekday(value: dt.date) -> float:
    return float(value.weekday())


def weekend_ordinal(value: dt.date) -> float:
    return 1.0 if value.weekday() >= 5 else 0.0


@lru_cache(maxsize=4096)
def classify_billing_day(value: dt.date) -> str:
    d = value.day
    total = days_in_month(value.year, value.month)
    if d <= 3 or d >= total - 2:
        return "month_boundary"
    if 4 <= d <= 10:
        return "early"
    if 11 <= d <= 20:
        return "mid"
    return "late"


def quarter_bucket(value: dt.date) -> int:
    return (value.month - 1) // 3


def seasonality_sin(value: dt.date, period_days: float) -> float:
    return float(np.sin(2.0 * np.pi * value.toordinal() / period_days))


def seasonality_cos(value: dt.date, period_days: float) -> float:
    return float(np.cos(2.0 * np.pi * value.toordinal() / period_days))
