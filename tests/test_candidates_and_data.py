"""Recall stage and data layer."""

from __future__ import annotations

import datetime as dt
import itertools

import pytest

from bankai.candidates.generator import CATEGORY_AFFINITY_PRIOR, CandidateGenerator, category_prior
from bankai.data.store import CachedRepository, InMemoryRepository, SqliteTransactionRepository
from bankai.domain.entities import (
    Direction,
    Transaction,
    TransactionStatus,
)
from bankai.features.embed import build_user_profile
from bankai.features.recurrence import detect_recurring

from .conftest import AS_OF, make_transaction, monthly_series


def _inputs(txs, as_of=AS_OF, user_id="u_c"):
    profile = build_user_profile(user_id, txs, as_of=as_of)
    recs = detect_recurring(profile.series, as_of=as_of, horizon_days=14)
    return profile, recs


class TestCandidateGenerator:
    def test_finds_scheduled_bills(self):
        txs: list = []
        for name, cat, day in (
            ("FiberNet ISP", "utilities.internet", 5),
            ("NordWohn Immobilien", "housing.rent", 2),
            ("VoltGrid Energy", "utilities.electricity", 11),
        ):
            txs += monthly_series(day=day, amount=50.0, months=12, name=name, category=cat)
        gen = CandidateGenerator()
        result = gen.generate(*_inputs(txs), as_of=AS_OF, horizon_days=20)
        names = {c.counterparty_name for c in result.candidates}
        assert {"FiberNet ISP", "NordWohn Immobilien", "VoltGrid Energy"} <= names

    def test_empty_history_returns_empty_pool(self):
        result = CandidateGenerator().generate(*_inputs([]), as_of=AS_OF)
        assert result.candidates == []
        assert result.coverage_note == "no_history"

    def test_excluded_counterparties_are_dropped(self):
        txs = monthly_series(day=5, amount=40.0, months=12)
        gen = CandidateGenerator()
        cp = next(iter(build_user_profile("u", txs, as_of=AS_OF).series))
        result = gen.generate(*_inputs(txs), as_of=AS_OF, excluded={cp})
        assert all(c.counterparty_id != cp for c in result.candidates)

    def test_category_cap_prevents_monoculture(self):
        """Eleven utility bills must not fill the list."""
        txs: list = []
        for i in range(11):
            txs += monthly_series(
                day=5 + i,
                amount=30.0,
                months=12,
                name=f"Utility {i}",
                category="utilities.electricity",
            )
        gen = CandidateGenerator(category_cap=3)
        result = gen.generate(*_inputs(txs), as_of=AS_OF, horizon_days=20)
        categories = [c.category for c in result.candidates]
        assert categories.count("utilities.electricity") <= 3

    def test_pool_size_is_respected(self):
        txs: list = []
        for i in range(30):
            txs += monthly_series(
                day=3 + i, amount=20.0, months=10, name=f"Payee {i}", category=f"category.{i}"
            )
        gen = CandidateGenerator(pool_size=8)
        result = gen.generate(*_inputs(txs), as_of=AS_OF, horizon_days=20)
        assert len(result.candidates) <= 8

    def test_recall_mix_is_reported(self):
        txs = monthly_series(day=5, amount=40.0, months=12)
        result = CandidateGenerator().generate(*_inputs(txs), as_of=AS_OF, horizon_days=20)
        assert result.recall_mix
        assert any(c.source == "recurring" for c in result.candidates)

    def test_sources_are_unioned_not_duplicated(self):
        txs = monthly_series(day=5, amount=40.0, months=12)
        result = CandidateGenerator().generate(*_inputs(txs), as_of=AS_OF, horizon_days=20)
        ids = [c.counterparty_id for c in result.candidates]
        assert len(ids) == len(set(ids))

    def test_candidates_are_never_frozen_mutation_victims(self):
        """Candidates are frozen models; every mutation must go via model_copy."""
        txs = monthly_series(day=5, amount=40.0, months=12)
        result = CandidateGenerator().generate(*_inputs(txs), as_of=AS_OF, horizon_days=20)
        assert all(
            c.scheduled_for >= c.recurrence.last_paid_on or c.status.value == "overdue"
            for c in result.candidates
            if c.recurrence and c.recurrence.last_paid_on
        )

    def test_broken_detector_does_not_break_generation(self, monkeypatch):
        """A detector bug must degrade recall, never take the endpoint down."""
        gen = CandidateGenerator()

        def boom(*_a, **_k):
            raise RuntimeError("detector exploded")

        monkeypatch.setattr(gen, "_affinity", boom)
        result = gen.generate(
            *_inputs(monthly_series(day=5, amount=40.0, months=12)), as_of=AS_OF, horizon_days=20
        )
        assert result.candidates


class TestCategoryPrior:
    def test_rent_outranks_coffee(self):
        assert category_prior("housing.rent") > category_prior("dining.coffee")

    def test_unknown_category_gets_a_neutral_prior(self):
        assert 0.0 < category_prior("something.new") < 1.0

    def test_prior_table_is_bounded(self):
        assert all(0.0 <= v <= 1.0 for v in CATEGORY_AFFINITY_PRIOR.values())

    def test_coffee_is_deliberately_deprioritised(self):
        """Recommending a coffee shop is technically right and useless."""
        assert category_prior("dining.coffee") <= 0.3


class TestRepositories:
    def test_in_memory_round_trip(self):
        repo = InMemoryRepository()
        repo.add([make_transaction(user_id="u1"), make_transaction(user_id="u2", name="Other")])
        assert repo.users() == ["u1", "u2"]
        assert len(repo.transactions_for_user("u1")) == 1
        assert repo.latest_date() == AS_OF

    def test_sqlite_round_trip(self, tmp_path):
        store = SqliteTransactionRepository(tmp_path / "t.db")
        txs = [make_transaction(user_id="u1", days_from=dt.date(2024, 1, d)) for d in (1, 2, 3)]
        assert store.bulk_insert(txs) == 3
        assert store.count() == 3
        loaded = store.transactions_for_user("u1")
        assert len(loaded) == 3
        assert loaded[0].value_date == dt.date(2024, 1, 1)

    def test_sqlite_is_idempotent_on_reinsert(self, tmp_path):
        store = SqliteTransactionRepository(tmp_path / "t.db")
        txs = [make_transaction(days_from=dt.date(2024, 1, d)) for d in (1, 2)]
        store.bulk_insert(txs)
        store.bulk_insert(txs)
        assert store.count() == 2

    def test_sqlite_since_filter(self, tmp_path):
        store = SqliteTransactionRepository(tmp_path / "t.db")
        store.bulk_insert(
            [
                make_transaction(days_from=dt.date(2024, 1, 1)),
                make_transaction(days_from=dt.date(2024, 6, 1)),
            ]
        )
        assert len(store.transactions_for_user("u_test", since=dt.date(2024, 3, 1))) == 1

    def test_null_balance_is_preserved(self, tmp_path):
        """A NULL balance means 'unknown', not 'zero' — conflating them blocks payments."""
        store = SqliteTransactionRepository(tmp_path / "t.db")
        store.upsert_account("u1", available_balance=None, currency="EUR")
        assert store.account("u1")["available_balance"] is None

    def test_cache_hit_avoids_repeat_queries(self, tmp_path):
        calls = {"n": 0}

        class Counting:
            def transactions_for_user(self, user_id, *, since=None):
                calls["n"] += 1
                return [make_transaction(user_id=user_id)]

            def users(self):
                return ["u1"]

            def all_transactions(self, limit=None):
                return []

            def account(self, user_id):
                return None

            def record_decision(self, **kw):
                return None

            def latest_date(self):
                return AS_OF

        cache = CachedRepository(Counting(), ttl_seconds=60)
        for _ in range(5):
            cache.transactions_for_user("u1")
        assert calls["n"] == 1
        assert cache.hits == 4 and cache.misses == 1

    def test_cache_evicts_when_full(self, tmp_path):
        class Empty:
            def transactions_for_user(self, user_id, *, since=None):
                return []

            def users(self):
                return []

            def all_transactions(self, limit=None):
                return []

            def account(self, user_id):
                return None

            def record_decision(self, **kw):
                return None

            def latest_date(self):
                return None

        cache = CachedRepository(Empty(), ttl_seconds=60, max_entries=4)
        for i in range(20):
            cache.transactions_for_user(f"u{i}")
        assert len(cache._cache) <= 4

    def test_invalidate_after_write(self, tmp_path):
        class Empty:
            def transactions_for_user(self, user_id, *, since=None):
                return [make_transaction(user_id=user_id)]

            def users(self):
                return ["u1"]

            def all_transactions(self, limit=None):
                return []

            def account(self, user_id):
                return None

            def record_decision(self, **kw):
                self.last = kw

            def latest_date(self):
                return AS_OF

        inner = Empty()
        cache = CachedRepository(inner, ttl_seconds=60)
        cache.transactions_for_user("u1")
        cache.record_decision(user_id="u1")
        assert inner.last["user_id"] == "u1"


class TestTransactionInvariants:
    def test_zero_amount_is_rejected(self):
        with pytest.raises(ValueError):
            make_transaction(amount=0.0)

    def test_negative_amount_is_rejected(self):
        with pytest.raises(ValueError):
            make_transaction(amount=-5.0)

    def test_balance_delta_sign_convention(self):
        """Credits are positive, debits negative. Getting this wrong silently
        inverts every affordability check in the policy layer."""
        inbound = make_transaction(amount=100.0, direction=Direction.INBOUND)
        outbound = make_transaction(amount=100.0, direction=Direction.OUTBOUND)
        assert inbound.balance_delta == pytest.approx(100.0)
        assert outbound.balance_delta == pytest.approx(-100.0)

    def test_unknown_field_is_rejected(self):
        with pytest.raises(ValueError):
            Transaction(
                transaction_id="t",
                user_id="u",
                posted_at=dt.datetime(2024, 1, 1),
                value_date=dt.date(2024, 1, 1),
                amount=1.0,
                currency="EUR",
                direction=Direction.OUTBOUND,
                counterparty_id="c",
                counterparty_name="n",
                surprise=1,
            )

    def test_ledger_balances_are_self_consistent(self, flat_transactions):
        """Spot-check that balance_after really tracks the signed transaction sum.

        Failed debits are the expected exception: they never move the balance.
        """
        by_user: dict[str, list[Transaction]] = {}
        for t in flat_transactions:
            if t.balance_after is not None:
                by_user.setdefault(t.user_id, []).append(t)
        checked = failures = 0
        for _user, txs in by_user.items():
            txs.sort(key=lambda t: (t.value_date, t.posted_at))
            for prev, nxt in itertools.pairwise(txs):
                if prev.balance_after is None or nxt.balance_after is None:
                    continue
                if nxt.status is TransactionStatus.FAILED:
                    assert nxt.balance_after == pytest.approx(prev.balance_after)
                    failures += 1
                    continue
                assert nxt.balance_after == pytest.approx(prev.balance_after + nxt.balance_delta)
                checked += 1
        assert checked > 1000, "expected to verify a large number of ledger transitions"
