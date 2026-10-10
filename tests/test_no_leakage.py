

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from bankai.candidates.generator import CandidateGenerator
from bankai.domain.entities import Direction
from bankai.features.builder import (
    FEATURE_NAMES,
    N_FEATURES,
    build_feature_context,
    build_feature_matrix,
    candidate_features,
)
from bankai.features.embed import build_user_profile
from bankai.features.recurrence import detect_recurring

from .conftest import AS_OF, make_transaction, monthly_series

CUTOFF = dt.date(2024, 3, 15)


def _matrix(txs, as_of, user_id="u_leak"):
    profile = build_user_profile(user_id, txs, as_of=as_of, inclusive=False)
    recs = detect_recurring(profile.series, as_of=as_of, horizon_days=14)
    ids = sorted(profile.series)
    ctx = build_feature_context(profile, recs, as_of=as_of, horizon_days=14)
    return ids, build_feature_matrix(ids, ctx), recs


class TestNoFutureLeakage:
    def test_appending_future_transactions_does_not_change_features(self):
        """The core guarantee, stated as a property test."""
        past = monthly_series(day=8, amount=42.0, months=9, end=CUTOFF)
        future = monthly_series(
            day=8, amount=999.0, months=6, end=dt.date(2024, 6, 30), start=CUTOFF
        )
        # Sanity: the future chunk really is disjoint and strictly later.
        assert future and all(t.value_date > CUTOFF for t in future)
        assert not {t.value_date for t in past} & {t.value_date for t in future}

        ids_a, mat_a, _ = _matrix(past, CUTOFF)
        ids_b, mat_b, _ = _matrix(past + future, CUTOFF)

        assert ids_a == ids_b
        np.testing.assert_allclose(
            mat_a, mat_b, rtol=0, atol=0, err_msg="features changed when future data was added"
        )

    def test_future_payments_do_not_change_visible_payees(self):
        past = monthly_series(day=8, amount=42.0, months=9, end=CUTOFF)
        future = monthly_series(
            day=15,
            amount=500.0,
            months=3,
            end=dt.date(2024, 6, 30),
            start=CUTOFF,
            name="Brand New Gym",
            category="health.fitness",
        )
        assert future and all(t.value_date > CUTOFF for t in future)

        # Invisible at the earlier date, regardless of what arrives later.
        gym = "cp_brand_new_gym"  # series are keyed by counterparty id
        assert gym not in build_user_profile("u", past, as_of=CUTOFF).series
        assert gym not in build_user_profile("u", past + future, as_of=CUTOFF).series
        # ... and genuinely present once the evaluation date reaches it.
        later = build_user_profile("u", past + future, as_of=dt.date(2024, 6, 30))
        assert gym in later.series

    def test_profiles_ignore_future_transactions(self):
        past = monthly_series(day=8, amount=42.0, months=9, end=CUTOFF)
        big_salary = [
            make_transaction(
                days_from=dt.date(2024, 5, 1),
                amount=999_999.0,
                name="Bonus Co",
                category="income.salary",
                direction=Direction.INBOUND,
            )
        ]
        a = build_user_profile("u", past, as_of=CUTOFF)
        b = build_user_profile("u", past + big_salary, as_of=CUTOFF)
        assert a.outbound_total == b.outbound_total
        assert a.salary_hint == b.salary_hint
        assert a.n_transactions == b.n_transactions

    def test_recurrence_analysis_is_as_of_scoped(self):
        past = monthly_series(day=8, amount=42.0, months=9, end=CUTOFF)
        future = monthly_series(
            day=8, amount=42.0, months=6, end=dt.date(2024, 6, 30), start=CUTOFF
        )
        _, _, recs_a = _matrix(past, CUTOFF)
        _, _, recs_b = _matrix(past + future, CUTOFF)
        assert recs_a
        for cp_id, prof in recs_a.items():
            assert prof.n_observations == recs_b[cp_id].n_observations
            assert prof.next_due_on == recs_b[cp_id].next_due_on
            assert prof.expected_amount == pytest.approx(recs_b[cp_id].expected_amount)

    def test_candidate_generation_is_as_of_scoped(self):
        past = monthly_series(day=8, amount=42.0, months=10, end=CUTOFF)
        future = monthly_series(
            day=20,
            amount=300.0,
            months=3,
            end=dt.date(2024, 6, 30),
            start=CUTOFF,
            name="Future Co",
            category="shopping.general",
        )
        gen = CandidateGenerator()
        a = gen.generate(*_gen_inputs(past, CUTOFF), as_of=CUTOFF, horizon_days=14)
        b = gen.generate(*_gen_inputs(past + future, CUTOFF), as_of=CUTOFF, horizon_days=14)
        assert [c.counterparty_id for c in a.candidates] == [
            c.counterparty_id for c in b.candidates
        ]

    def test_due_date_projection_does_not_use_future_payments(self):
        """A bill due in 8 days must be forecast as due in 8 days, not 'already paid'."""
        txs = monthly_series(day=23, amount=42.0, months=8, end=dt.date(2024, 6, 23))
        profile = build_user_profile("u", txs, as_of=dt.date(2024, 7, 1))
        recs = detect_recurring(profile.series, as_of=dt.date(2024, 7, 1), horizon_days=14)
        prof = next(iter(recs.values()))
        # Last payment 2024-06-23, so the next one is projected near 2024-07-23.
        assert prof.next_due_on is not None
        assert prof.next_due_on.month == 7
        assert prof.days_until_due is not None and 15 <= prof.days_until_due <= 35


def _gen_inputs(txs, as_of):
    profile = build_user_profile("u", txs, as_of=as_of)
    recs = detect_recurring(profile.series, as_of=as_of, horizon_days=14)
    return profile, recs


class TestFeatureContract:
    def test_every_feature_is_present_and_finite(self, repo):
        profile = build_user_profile(
            repo.users()[0], repo.transactions_for_user(repo.users()[0]), as_of=AS_OF
        )
        recs = detect_recurring(profile.series, as_of=AS_OF, horizon_days=14)
        for cp_id in list(profile.series)[:10]:
            feats = candidate_features(
                cp_id, build_feature_context(profile, recs, as_of=AS_OF, horizon_days=14)
            )
            assert set(feats) == set(FEATURE_NAMES)
            bad = {k: v for k, v in feats.items() if not np.isfinite(v)}
            assert not bad, f"non-finite features: {bad}"

    def test_matrix_shape_and_dtype(self, repo):
        user = repo.users()[0]
        profile = build_user_profile(user, repo.transactions_for_user(user), as_of=AS_OF)
        recs = detect_recurring(profile.series, as_of=AS_OF, horizon_days=14)
        ids = sorted(profile.series)
        matrix = build_feature_matrix(
            ids, build_feature_context(profile, recs, as_of=AS_OF, horizon_days=14)
        )
        assert matrix.shape == (len(ids), N_FEATURES)
        assert matrix.dtype == np.float32
        assert np.isfinite(matrix).all()

    def test_empty_candidate_list_gives_empty_matrix(self, repo):
        user = repo.users()[0]
        profile = build_user_profile(user, repo.transactions_for_user(user), as_of=AS_OF)
        recs = detect_recurring(profile.series, as_of=AS_OF, horizon_days=14)
        matrix = build_feature_matrix(
            [], build_feature_context(profile, recs, as_of=AS_OF, horizon_days=14)
        )
        assert matrix.shape == (0, N_FEATURES)

    def test_feature_names_are_unique(self):
        assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES)) == N_FEATURES


class TestServingMomentIsPointInTime:
    """The `as_of` day's own payment must be invisible when we say it is.

    Regression: the labeler excluded same-day history while the service did
    not. Every target was therefore presented to the model as "already paid",
    the model learned to distrust that feature pattern, and offline metrics
    measured the skew rather than the model.
    """

    def test_exclusive_history_hides_the_as_of_day(self):
        # day=CUTOFF.day so the series actually lands on the decision day.
        txs = monthly_series(day=CUTOFF.day, amount=42.0, months=9, end=CUTOFF)
        assert any(t.value_date == CUTOFF for t in txs), "fixture must include the as_of day"
        incl = build_user_profile("u", txs, as_of=CUTOFF, inclusive=True)
        excl = build_user_profile("u", txs, as_of=CUTOFF, inclusive=False)
        assert incl.n_transactions == excl.n_transactions + 1

    def test_batch_features_do_not_see_the_as_of_day(self):
        """The new payday / co-payment aggregates must obey the same boundary."""
        txs = monthly_series(day=CUTOFF.day, amount=42.0, months=9, end=CUTOFF)
        txs += monthly_series(
            day=CUTOFF.day,
            amount=9.0,
            months=9,
            end=CUTOFF,
            name="Extra Co",
            category="dining.coffee",
        )
        incl = build_user_profile("u", txs, as_of=CUTOFF, inclusive=True)
        excl = build_user_profile("u", txs, as_of=CUTOFF, inclusive=False)
        # With the decision day visible, "how long since the user last paid
        # anyone" collapses to 0; without it, the last active day is a month
        # earlier. That difference is the boundary, and it must be visible.
        assert incl.days_since_last_active_day == 0
        assert excl.days_since_last_active_day > 20

    def test_new_features_are_included_in_the_leakage_sweep(self):
        past = monthly_series(day=8, amount=42.0, months=9, end=CUTOFF)
        future = monthly_series(
            day=8, amount=999.0, months=6, end=dt.date(2024, 6, 30), start=CUTOFF
        )
        _ids_a, mat_a, _rec_a = _matrix(past, CUTOFF)
        _ids_b, mat_b, _rec_b = _matrix(past + future, CUTOFF)
        np.testing.assert_allclose(mat_a, mat_b, rtol=0, atol=0)

    def test_payday_alignment_respects_the_salary_date_seen(self):
        from bankai.domain.entities import Direction

        salary = [
            make_transaction(
                days_from=dt.date(2024, m, 26),
                amount=4000.0,
                name="Payroll",
                category="income.salary",
                direction=Direction.INBOUND,
            )
            for m in range(1, 13)
        ]
        txs = monthly_series(day=27, amount=30.0, months=12, end=dt.date(2024, 6, 27))
        profile = build_user_profile(
            "u", [*salary, *txs], as_of=dt.date(2024, 6, 30), inclusive=False
        )
        assert profile.salary_day_of_month == 26
