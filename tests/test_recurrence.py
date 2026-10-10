

from __future__ import annotations

import datetime as dt

import pytest

from bankai.domain.entities import RecurrenceStatus
from bankai.features.embed import build_user_profile
from bankai.features.recurrence import (
    analyse_series,
    detect_recurring,
    discover_cycles,
    fit_period,
)
from bankai.features.series import build_series

from .conftest import AS_OF, make_transaction, monthly_series


def _series_of(txs):
    """Single-series view of a transaction list."""
    return next(iter(build_series(txs).values()))


# --------------------------------------------------------------- period fitting
class TestPeriodFitting:
    def test_pure_monthly(self):
        dates = [dt.date(2024, m, 5) for m in range(1, 13)]
        fit = fit_period(dates)
        assert 25.0 <= fit.period <= 33.0
        assert fit.coverage > 0.9

    def test_weekly(self):
        dates = [dt.date(2024, 1, 1) + dt.timedelta(days=7 * i) for i in range(15)]
        assert 6.0 <= fit_period(dates).period <= 8.0

    def test_quarterly(self):
        dates = [dt.date(2023, m, 10) for m in (1, 4, 7, 10)] + [dt.date(2024, 1, 10)]
        assert 80.0 <= fit_period(dates).period <= 100.0

    def test_single_day_cadence_is_not_a_degenerate_winner(self):
        """Regression: period 1.0 with unbounded multiples fits any series exactly.

        A 30-day gap explained as 30 one-day cycles has zero residual and won
        every series. Multiple-count is now capped, so monthly is recovered.
        """
        dates = [dt.date(2024, m, 3) for m in range(1, 13)]
        assert fit_period(dates).period > 20.0

    def test_calendar_aware_not_day_counting(self):
        """Regression: unequal month lengths made a 30-day bill look like 35 days.

        Billing on the 12th gives 28/29/30/31-day gaps with zero jitter. Only a
        calendar-aware prediction recovers "monthly".
        """
        dates = [
            dt.date(2023, 11, 12),
            dt.date(2023, 12, 12),
            dt.date(2024, 1, 12),
            dt.date(2024, 2, 12),
            dt.date(2024, 3, 12),
            dt.date(2024, 4, 12),
        ]
        fit = fit_period(dates)
        assert 25.0 <= fit.period <= 33.0, f"expected monthly, got {fit.period}"

    def test_day_of_month_29_and_30_31_are_all_monthly(self):
        for day in (1, 15, 28, 29, 30, 31):
            dates: list[dt.date] = []
            y, m = 2023, 1
            for _ in range(14):
                import calendar

                d = min(day, calendar.monthrange(y, m)[1])
                dates.append(dt.date(y, m, d))
                m += 1
                if m > 12:
                    y, m = y + 1, 1
            assert 25.0 <= fit_period(dates).period <= 33.0, f"day {day} misidentified"

    def test_skipped_cycles_are_absorbed(self):
        dates = [
            dt.date(2024, 1, 5),
            dt.date(2024, 2, 5),
            dt.date(2024, 5, 5),
            dt.date(2024, 6, 5),
            dt.date(2024, 7, 5),
            dt.date(2024, 8, 5),
        ]
        fit = fit_period(dates)
        assert 25.0 <= fit.period <= 33.0
        # The 3-month gap is explained as three monthly cadences, not a 90-day one.
        assert fit.coverage > 0.95
        assert fit.absolute_error < 2.0
        assert fit.mean_skipped_cycles > 0.3

    def test_very_irregular_series_yields_no_cadence(self):
        rng = __import__("numpy").random.default_rng(1)
        base = dt.date(2022, 1, 1)
        dates = sorted({base + dt.timedelta(days=int(rng.integers(0, 1500))) for _ in range(20)})
        assert fit_period(dates).coverage < 0.6

    def test_too_few_dates(self):
        assert fit_period([dt.date(2024, 1, 1)]).coverage == 0.0
        assert fit_period([]).coverage == 0.0

    def test_duplicate_dates_do_not_break_fitting(self):
        dates = [dt.date(2024, m, 7) for m in range(1, 13)] * 2
        assert 25.0 <= fit_period(dates).period <= 33.0


# ------------------------------------------------------------------ analysis
class TestAnalyseSeries:
    def test_confident_monthly_bill(self):
        txs = monthly_series(day=5, amount=40.0, months=14)
        profile = analyse_series(_series_of(txs), as_of=AS_OF, horizon_days=14)
        assert profile.is_recurring
        assert 25.0 <= (profile.period_days or 0) <= 33.0
        assert profile.confidence > 0.5
        assert profile.n_observations >= 10
        assert profile.amount_stability > 0.8
        assert profile.adherence > 0.8
        assert profile.next_due_on is not None
        assert profile.expected_amount == pytest.approx(40.0, rel=0.05)

    def test_jittered_series_still_recurring(self):
        txs = monthly_series(day=12, amount=55.0, months=16, jitter_days=3, amount_noise=0.04)
        profile = analyse_series(_series_of(txs), as_of=AS_OF)
        assert profile.is_recurring
        assert profile.amount_stability > 0.5

    def test_single_transaction_is_not_recurring(self):
        txs = [make_transaction(days_from=dt.date(2024, 6, 1))]
        profile = analyse_series(_series_of(txs), as_of=AS_OF)
        assert not profile.is_recurring
        assert profile.status is RecurrenceStatus.ONE_OFF
        assert profile.n_observations == 1

    def test_abandoned_subscription_is_not_flagged_overdue(self):
        """Regression: a relationship with 2+ silent cycles is not 'overdue'.

        A cancelled subscription was being surfaced as an overdue bill, which is
        the most damaging false positive this system can make.
        """
        old = monthly_series(day=10, amount=9.99, months=8, end=dt.date(2023, 12, 31))
        profile = analyse_series(_series_of(old), as_of=AS_OF, horizon_days=14)
        assert not profile.is_recurring
        assert profile.status not in (
            RecurrenceStatus.OVERDUE,
            RecurrenceStatus.DUE_NOW,
            RecurrenceStatus.DUE_SOON,
        )

    def test_single_missed_cycle_is_flagged_overdue(self):
        """The useful half of the rule: one silent cycle *is* an unpaid bill."""
        txs = monthly_series(day=10, amount=25.0, months=10, end=dt.date(2024, 5, 10))
        profile = analyse_series(_series_of(txs), as_of=AS_OF, horizon_days=14)
        assert profile.is_recurring
        assert profile.missed_cycles == 1
        assert profile.status is RecurrenceStatus.OVERDUE

    def test_status_due_now(self):
        txs = monthly_series(day=AS_OF.day, amount=20.0, months=6, end=dt.date(2024, 6, 25))
        profile = analyse_series(_series_of(txs), as_of=AS_OF, horizon_days=14)
        assert profile.status in (RecurrenceStatus.DUE_NOW, RecurrenceStatus.DUE_SOON)

    def test_status_recently_paid(self):
        """Paid within the last week, next due well outside the window."""
        txs = monthly_series(day=1, amount=20.0, months=14, end=dt.date(2024, 6, 3))
        profile = analyse_series(_series_of(txs), as_of=dt.date(2024, 6, 8), horizon_days=14)
        assert profile.status is RecurrenceStatus.RECENTLY_PAID
        assert profile.days_until_due is not None and profile.days_until_due > 14

    def test_amount_range_brackets_estimate(self):
        txs = monthly_series(day=8, amount=100.0, months=10, amount_noise=0.12)
        profile = analyse_series(_series_of(txs), as_of=AS_OF)
        assert profile.amount_low is not None and profile.amount_high is not None
        assert profile.amount_low <= profile.expected_amount <= profile.amount_high

    def test_confidence_is_zero_when_not_recurring(self):
        txs = [make_transaction(days_from=dt.date(2024, 6, 1))]
        assert analyse_series(_series_of(txs), as_of=AS_OF).confidence == 0.0


# ------------------------------------------------------------------- bundles
class TestBundleDiscovery:
    def test_groups_payees_due_on_the_same_day(self):
        txs: list = []
        for name, cat in (
            ("StreamBox TV", "entertainment.tv"),
            ("Nordic Telecom Mobile", "utilities.phone"),
            ("HomeShield Security", "utilities.security"),
        ):
            txs += monthly_series(day=14, amount=30.0, months=10, name=name, category=cat)
        series = build_series(txs)
        recs = detect_recurring(series, as_of=AS_OF, horizon_days=20)
        bundles = discover_cycles(recs)
        assert bundles, "expected a bundle of co-scheduled payees"
        assert max(len(b) for b in bundles) >= 3

    def test_does_not_group_differently_scheduled_payees(self):
        txs: list = []
        txs += monthly_series(day=2, amount=30.0, months=10, name="Rent Co")
        txs += monthly_series(
            day=20, amount=30.0, months=10, name="Gym Ltd", category="health.fitness"
        )
        recs = detect_recurring(build_series(txs), as_of=AS_OF, horizon_days=20)
        assert all(len(b) < 3 for b in discover_cycles(recs))


# ---------------------------------------------- accuracy against ground truth
def test_recall_and_precision_against_generator_ground_truth(flat_transactions, corpus):
    """End-to-end check: does schedule detection match how the data was made?"""
    by_user, _ = corpus
    truth: dict[str, str] = {}
    from bankai.data.synthetic import generate_dataset

    _txs, blueprints, _end = generate_dataset(n_users=12, months=18, as_of=AS_OF, seed=17)
    for bp in blueprints:
        for name, _cat, _amt, cadence, _day in bp.obligations:
            truth[name] = "quarterly" if cadence == "quarterly" else "monthly"
        truth[bp.landlord] = "monthly"
        for name, _cat, _amt, cadence, _day in bp.discretionary:
            if cadence in {"weekly", "monthly"}:
                truth[name] = cadence

    detected = correct = total_true = 0
    for user_id, txs in by_user.items():
        profile = build_user_profile(user_id, txs, as_of=AS_OF)
        for rec in detect_recurring(profile.series, as_of=AS_OF, horizon_days=14).values():
            expected = truth.get(rec.counterparty_name)
            if expected is None:
                continue
            total_true += 1
            if not rec.is_recurring:
                continue
            detected += 1
            period = rec.period_days or 0
            pred = "weekly" if period < 14 else "quarterly" if period > 60 else "monthly"
            correct += pred == expected

    assert total_true > 60, "corpus too small to be meaningful"
    assert detected / total_true > 0.85, f"schedule recall too low: {detected}/{total_true}"
    assert correct / detected > 0.90, f"cadence precision too low: {correct}/{detected}"


class TestEvidenceVersusActionability:
    """Regression: a dormant-but-obvious pattern was reported as a one-off.

    A weekly transit pass paid 47 times, unused for three weeks, came back as
    ``status=one_off, confidence=0.00``. Two distinct questions were being
    conflated: *how strong is the evidence* and *should we recommend it now*.
    """

    def test_dormant_pattern_keeps_its_status_and_evidence(self):
        # 47 weekly payments, the last landing three weeks before `as_of`: a
        # pattern the customer has clearly stopped using.
        last = dt.date(2024, 6, 9)
        weekly = [
            make_transaction(
                days_from=last - dt.timedelta(days=7 * (46 - i)),
                amount=6.40,
                name="FareGo Transit",
                category="transport.fare",
            )
            for i in range(47)
        ]
        profile = analyse_series(_series_of(weekly), as_of=AS_OF, horizon_days=14)

        assert 6.0 <= (profile.period_days or 0) <= 8.0, "weekly cadence should still be found"
        assert not profile.is_recurring, "three silent weeks means it is not actionable"
        assert profile.status is not RecurrenceStatus.ONE_OFF
        assert profile.confidence > 0.0, "evidence strength is not zero"
        assert profile.actionable_confidence == 0.0, "but it must not be recommendable"

    def test_evidence_is_not_gated_on_actionability(self):
        live = monthly_series(day=5, amount=40.0, months=14)
        profile = analyse_series(_series_of(live), as_of=AS_OF, horizon_days=14)
        assert profile.confidence == pytest.approx(profile.actionable_confidence)

    def test_single_payment_has_no_evidence(self):
        txs = [make_transaction(days_from=dt.date(2024, 6, 1))]
        profile = analyse_series(_series_of(txs), as_of=AS_OF)
        assert profile.confidence == 0.0
        assert profile.actionable_confidence == 0.0

    def test_abandoned_subscription_reports_its_skipped_cycles(self):
        """Missed-cycle counting must not depend on the recurring flag."""
        old = monthly_series(day=10, amount=9.99, months=8, end=dt.date(2023, 12, 31))
        profile = analyse_series(_series_of(old), as_of=AS_OF, horizon_days=14)
        assert profile.missed_cycles >= 2
        assert profile.status is RecurrenceStatus.RECURRING
        assert profile.next_due_on == AS_OF, "no meaningful due date once a pattern is dormant"
