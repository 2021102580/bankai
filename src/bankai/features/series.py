

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np

from bankai.domain.entities import Transaction, TransactionStatus
from bankai.features.calendar import (
    MONTH_MEAN_DAYS,
    bounded,
    classify_billing_day,
    month_key,
    robust_cv,
)


@dataclass(frozen=True, slots=True)
class SeriesPoint:
    value_date: dt.date
    amount: float
    status: str = TransactionStatus.POSTED.value
    channel: str = "unknown"
    is_failed: bool = False


@dataclass
class CounterpartySeries:
    counterparty_id: str
    counterparty_name: str
    category: str
    currency: str
    points: list[SeriesPoint] = field(default_factory=list)

    # ------------------------------------------------------------------ views
    @property
    def dates(self) -> list[dt.date]:
        return [p.value_date for p in self.points]

    @property
    def amounts(self) -> np.ndarray:
        return np.asarray([p.amount for p in self.points], dtype=float)

    @property
    def ordinals(self) -> np.ndarray:
        return np.asarray([p.value_date.toordinal() for p in self.points], dtype=float)

    @property
    def n(self) -> int:
        return len(self.points)

    @property
    def first_date(self) -> dt.date | None:
        return self.points[0].value_date if self.points else None

    @property
    def last_date(self) -> dt.date | None:
        return self.points[-1].value_date if self.points else None

    @property
    def last_amount(self) -> float | None:
        return self.points[-1].amount if self.points else None

    @property
    def failed_count(self) -> int:
        return sum(1 for p in self.points if p.is_failed)

    # -------------------------------------------------------------- summaries
    def median_amount(self, window: int | None = None) -> float | None:
        values = self.amounts[-window:] if window else self.amounts
        return float(np.median(values)) if values.size else None

    def mean_amount(self) -> float | None:
        return float(self.amounts.mean()) if self.n else None

    def amount_cv(self, window: int | None = None) -> float:
        values = self.amounts[-window:] if window else self.amounts
        return robust_cv(values)

    def total_amount(self) -> float:
        return float(self.amounts.sum()) if self.n else 0.0

    def count_in_window(self, as_of: dt.date, days: int) -> int:
        cutoff = (as_of - dt.timedelta(days=days)).toordinal()
        return sum(1 for p in self.points if p.value_date.toordinal() > cutoff)

    def amount_in_window(self, as_of: dt.date, days: int) -> float:
        cutoff = (as_of - dt.timedelta(days=days)).toordinal()
        return float(sum(p.amount for p in self.points if p.value_date.toordinal() > cutoff))

    def days_since_last(self, as_of: dt.date) -> int | None:
        last = self.last_date
        if last is None:
            return None
        return (as_of - last).days

    def days_between_points(self) -> np.ndarray:
        if self.n < 2:
            return np.empty(0, dtype=float)
        ords = self.ordinals
        return np.diff(ords)

    def distinct_months(self) -> int:
        return len({month_key(p.value_date) for p in self.points})

    def active_month_ratio(self, as_of: dt.date) -> float:
        """Share of the observation window in which this payee was used."""
        if self.n == 0:
            return 0.0
        first, last = self.first_date, self.last_date
        assert first is not None and last is not None
        span_months = (last.year - first.year) * 12 + (last.month - first.month) + 1
        return bounded(self.distinct_months() / max(span_months, 1))

    def monthly_amount_series(self) -> dict[tuple[int, int], float]:
        out: dict[tuple[int, int], float] = {}
        for p in self.points:
            key = month_key(p.value_date)
            out[key] = out.get(key, 0.0) + p.amount
        return out

    def mean_gap(self) -> float | None:
        gaps = self.days_between_points()
        return float(gaps.mean()) if gaps.size else None

    def dominant_billing_day(self) -> int | None:
        if self.n == 0:
            return None
        days = [p.value_date.day for p in self.points]
        counts: dict[int, int] = {}
        for d in days:
            counts[d] = counts.get(d, 0) + 1
        best = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))
        return int(best[0])

    def billing_day_histogram(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for p in self.points:
            counts[p.value_date.day] = counts.get(p.value_date.day, 0) + 1
        return counts

    def billing_day_class_mode(self) -> str | None:
        if self.n == 0:
            return None
        classes: dict[str, int] = {}
        for p in self.points:
            c = classify_billing_day(p.value_date)
            classes[c] = classes.get(c, 0) + 1
        return max(classes.items(), key=lambda kv: (kv[1], kv[0]))[0]

    def typical_weekday(self) -> int | None:
        if self.n == 0:
            return None
        counts: dict[int, int] = {}
        for p in self.points:
            wd = p.value_date.weekday()
            counts[wd] = counts.get(wd, 0) + 1
        return int(max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0])

    def amount_trend_per_cycle(self, period: float) -> float:
        """Least-squares slope of amount against cycle index (e.g. annual rise)."""
        if self.n < 3 or period <= 0:
            return 0.0
        x = self.ordinals / period
        y = self.amounts
        x_mean, y_mean = x.mean(), y.mean()
        denom = float(((x - x_mean) ** 2).sum())
        if denom < 1e-9:
            return 0.0
        return float(((x - x_mean) * (y - y_mean)).sum() / denom)

    def expected_amount(
        self,
        period: float | None,
        *,
        as_of: dt.date | None = None,
        window: int = 6,
    ) -> float:
        if self.n == 0:
            return 0.0
        values = self.amounts[-window:]
        if values.size == 0:
            return 0.0
        median = float(np.median(values))
        recent = float(values[-1])
        blended = 0.55 * median + 0.45 * recent
        if period and period > 0 and self.n >= 4 and as_of is not None:
            slope = self.amount_trend_per_cycle(period)
            cycles_ahead = 1.0
            predicted = blended + slope * cycles_ahead
            blended = 0.5 * blended + 0.5 * float(np.clip(predicted, 0.5 * median, 2.0 * median))
        return float(max(round(blended, 2), 0.01))

    def amount_quantiles(self, window: int = 8) -> tuple[float, float]:
        values = self.amounts[-window:]
        if values.size == 0:
            return (0.0, 0.0)
        return float(np.quantile(values, 0.1)), float(np.quantile(values, 0.9))

    def text_tokens(self) -> list[str]:
        raw = f"{self.counterparty_name} {self.category}".lower()
        return [t for t in "".join(c if c.isalnum() else " " for c in raw).split() if len(t) > 1]


def build_series(transactions: list[Transaction]) -> dict[str, CounterpartySeries]:
    grouped: dict[str, list[Transaction]] = {}
    for tx in transactions:
        if tx.direction.value != "outbound":
            continue
        if tx.status is TransactionStatus.REVERSED:
            continue
        grouped.setdefault(tx.counterparty_id, []).append(tx)

    series: dict[str, CounterpartySeries] = {}
    for cp_id, txs in grouped.items():
        txs.sort(key=lambda t: (t.value_date, t.posted_at))
        head = txs[0]
        series[cp_id] = CounterpartySeries(
            counterparty_id=cp_id,
            counterparty_name=head.counterparty_name,
            category=head.counterparty_category,
            currency=head.currency,
            points=[
                SeriesPoint(
                    value_date=tx.value_date,
                    amount=tx.amount,
                    status=tx.status.value,
                    channel=tx.channel.value,
                    is_failed=tx.status is TransactionStatus.FAILED,
                )
                for tx in txs
            ],
        )
    return series


def coarsen(series: CounterpartySeries, minimum_amount: float = 1.0) -> CounterpartySeries | None:
    if series.n == 0:
        return None
    return series if series.median_amount() and series.median_amount() >= minimum_amount else None


__all__ = [
    "MONTH_MEAN_DAYS",
    "CounterpartySeries",
    "SeriesPoint",
    "build_series",
    "coarsen",
]
