

from __future__ import annotations

import datetime as dt

import pytest

from bankai.domain.entities import RecurrenceProfile, RecurrenceStatus, TransferCandidate
from bankai.ranking.policy import Decision, PolicyContext, PolicyEngine

AS_OF = dt.date(2024, 6, 30)


def make_candidate(
    amount: float = 25.0,
    cp_id: str = "cp_isp",
    status: RecurrenceStatus = RecurrenceStatus.DUE_NOW,
    scheduled_for: dt.date | None = None,
) -> TransferCandidate:
    return TransferCandidate(
        user_id="u_1",
        counterparty_id=cp_id,
        counterparty_name="FiberNet ISP",
        category="utilities.internet",
        suggested_amount=amount,
        currency="EUR",
        scheduled_for=scheduled_for or AS_OF,
        status=status,
        recurrence=RecurrenceProfile(
            counterparty_id=cp_id, is_recurring=True, periodicity_confidence=0.9
        ),
    )


def make_ctx(**overrides) -> PolicyContext:
    base = {
        "user_id": "u_1",
        "as_of": AS_OF,
        "available_balance": 5_000.0,
        "currency": "EUR",
    }
    base.update(overrides)
    return PolicyContext(**base)


@pytest.fixture
def engine() -> PolicyEngine:
    return PolicyEngine(
        max_single=1_000.0,
        daily_limit=2_000.0,
        monthly_limit=10_000.0,
        balance_buffer=50.0,
        min_amount=0.5,
        enable=True,
    )


class TestHardBlocks:
    def test_sanctioned_counterparty_is_blocked(self, engine):
        ctx = make_ctx(sanctioned_counterparties=frozenset({"cp_isp"}))
        verdict = engine.evaluate(make_candidate(), ctx)
        assert verdict.decision is Decision.BLOCK
        assert "sanctions_hit" in verdict.flags

    def test_sanctions_block_survives_fail_open(self):
        """A screening hit must never be downgraded to a warning."""
        engine = PolicyEngine(enable=True, fail_open=True)
        ctx = make_ctx(sanctioned_counterparties=frozenset({"cp_isp"}))
        assert engine.evaluate(make_candidate(), ctx).decision is Decision.BLOCK

    def test_user_blocklist_suppresses(self, engine):
        ctx = make_ctx(blocked_counterparties=frozenset({"cp_isp"}))
        verdict = engine.evaluate(make_candidate(), ctx)
        assert verdict.decision is Decision.SUPPRESS
        assert "user_blocked_payee" in verdict.flags


class TestAmounts:
    def test_below_minimum_is_suppressed(self, engine):
        assert engine.evaluate(make_candidate(0.1), make_ctx()).decision is Decision.SUPPRESS

    def test_above_single_transfer_limit(self, engine):
        verdict = engine.evaluate(make_candidate(5_000.0), make_ctx())
        assert verdict.decision is Decision.SUPPRESS
        assert "exceeds_single_transfer_limit" in verdict.flags

    def test_daily_limit_pressure(self, engine):
        ctx = make_ctx(daily_spent=1_990.0)
        verdict = engine.evaluate(make_candidate(25.0), ctx)
        assert verdict.decision is Decision.SUPPRESS
        assert "daily_limit_pressure" in verdict.flags

    def test_monthly_limit_pressure(self, engine):
        ctx = make_ctx(monthly_spent=9_990.0)
        verdict = engine.evaluate(make_candidate(25.0), ctx)
        assert verdict.decision is Decision.SUPPRESS
        assert "monthly_limit_pressure" in verdict.flags


class TestFunds:
    def test_sufficient_balance_passes(self, engine):
        assert engine.evaluate(make_candidate(100.0), make_ctx()).decision is Decision.ALLOW

    def test_insufficient_balance_suppresses(self, engine):
        ctx = make_ctx(available_balance=100.0)
        verdict = engine.evaluate(make_candidate(500.0), ctx)
        assert verdict.decision is Decision.SUPPRESS
        assert "insufficient_balance" in verdict.flags

    def test_buffer_is_respected(self, engine):
        """Balance exactly equal to the amount (plus buffer) is not spendable."""
        ctx = make_ctx(available_balance=70.0)
        assert engine.evaluate(make_candidate(25.0), ctx).decision is Decision.SUPPRESS

    def test_zero_balance_suppresses_everything(self, engine):
        verdict = engine.evaluate(make_candidate(1.0), make_ctx(available_balance=0.0))
        assert verdict.decision is Decision.SUPPRESS
        assert "no_estimated_balance" in verdict.flags

    def test_unknown_balance_does_not_invent_headroom(self, engine):
        """An unobservable balance must not silently unlock payments."""
        verdict = engine.evaluate(make_candidate(1.0), make_ctx(available_balance=0.0))
        assert "no_estimated_balance" in verdict.flags


class TestScheduling:
    def test_recently_paid_is_flagged_but_allowed(self, engine):
        cand = make_candidate(status=RecurrenceStatus.RECENTLY_PAID)
        verdict = engine.evaluate(cand, make_ctx())
        assert verdict.decision is Decision.ALLOW
        assert "already_paid_this_cycle" in verdict.flags

    def test_overdue_is_flagged_and_scheduled_today(self, engine):
        cand = make_candidate(status=RecurrenceStatus.OVERDUE, scheduled_for=AS_OF)
        verdict = engine.evaluate(cand, make_ctx())
        assert "overdue_suggested_today" in verdict.flags
        assert verdict.decision is Decision.ALLOW

    def test_after_cutoff_is_flagged(self, engine):
        cand = make_candidate(status=RecurrenceStatus.DUE_NOW, scheduled_for=AS_OF)
        ctx = make_ctx(is_cutoff_passed=True)
        assert "after_cutoff_next_day_estimate" in engine.evaluate(cand, ctx).flags


class TestRobustness:
    def test_disabled_policy_allows_everything(self):
        engine = PolicyEngine(enable=False)
        ctx = make_ctx(available_balance=0.0, sanctioned_counterparties=frozenset({"cp_isp"}))
        assert engine.evaluate(make_candidate(), ctx).decision is Decision.ALLOW

    def test_policy_error_fails_closed_by_default(self, engine, monkeypatch):
        def boom(*_a, **_k):
            raise RuntimeError("limits service down")

        monkeypatch.setattr(engine, "_evaluate", boom)
        assert engine.evaluate(make_candidate(), make_ctx()).decision is Decision.SUPPRESS

    def test_policy_error_can_fail_open(self, monkeypatch):
        engine = PolicyEngine(enable=True, fail_open=True)
        monkeypatch.setattr(
            engine, "_evaluate", lambda *a, **k: (_ for _ in ()).throw(RuntimeError())
        )
        verdict = engine.evaluate(make_candidate(), make_ctx())
        assert verdict.decision is Decision.ALLOW
        assert "policy_error_fail_open" in verdict.flags

    def test_batch_preserves_order(self, engine):
        cands = [make_candidate(10.0, f"cp_{i}") for i in range(5)]
        out = engine.batch(cands, make_ctx())
        assert [c.counterparty_id for c, _ in out] == [c.counterparty_id for c in cands]
