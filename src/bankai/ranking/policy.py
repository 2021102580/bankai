

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from enum import StrEnum

from bankai.domain.entities import RecurrenceStatus, TransferCandidate
from bankai.logging_setup import get_logger

log = get_logger(__name__)


class Decision(StrEnum):
    ALLOW = "allow"
    SUPPRESS = "suppress"
    BLOCK = "block"  # hard block, surfaced as an error to the client


@dataclass(frozen=True, slots=True)
class PolicyContext:
    user_id: str
    as_of: dt.date
    available_balance: float
    currency: str = "EUR"
    blocked_counterparties: frozenset[str] = field(default_factory=frozenset)
    sanctioned_counterparties: frozenset[str] = field(default_factory=frozenset)
    daily_spent: float = 0.0
    monthly_spent: float = 0.0
    daily_limit: float = 100_000.0
    monthly_limit: float = 500_000.0
    max_single: float = 100_000.0
    min_amount: float = 0.5
    cutoff_hour: int = 21
    is_cutoff_passed: bool = False
    fail_open: bool = False


@dataclass(frozen=True, slots=True)
class PolicyVerdict:
    decision: Decision
    flags: tuple[str, ...] = ()
    reason: str = ""


class PolicyEngine:

    def __init__(
        self,
        *,
        max_single: float = 100_000.0,
        daily_limit: float = 100_000.0,
        monthly_limit: float = 500_000.0,
        balance_buffer: float = 50.0,
        min_amount: float = 0.5,
        cutoff_hour: int = 21,
        enable: bool = True,
        fail_open: bool = False,
    ) -> None:
        self.max_single = max_single
        self.daily_limit = daily_limit
        self.monthly_limit = monthly_limit
        self.balance_buffer = balance_buffer
        self.min_amount = min_amount
        self.cutoff_hour = cutoff_hour
        self.enable = enable
        self.fail_open = fail_open

    def evaluate(self, cand: TransferCandidate, ctx: PolicyContext) -> PolicyVerdict:
        if not self.enable:
            return PolicyVerdict(Decision.ALLOW)
        try:
            return self._evaluate(cand, ctx)
        except Exception:  # pragma: no cover - a policy bug must not take down serving
            log.exception(
                "policy evaluation failed", extra={"counterparty_id": cand.counterparty_id}
            )
            return (
                PolicyVerdict(
                    Decision.ALLOW, flags=("policy_error_fail_open",), reason="policy error"
                )
                if self.fail_open
                else PolicyVerdict(
                    Decision.SUPPRESS, flags=("policy_error",), reason="policy error"
                )
            )

    def _evaluate(self, cand: TransferCandidate, ctx: PolicyContext) -> PolicyVerdict:
        cp_id = cand.counterparty_id
        amount = float(cand.suggested_amount)

        # --- hard blocks (compliance first, never fail open) ---------------
        if cp_id in ctx.sanctioned_counterparties:
            return PolicyVerdict(
                Decision.BLOCK, ("sanctions_hit",), "counterparty blocked by screening"
            )
        if cp_id in ctx.blocked_counterparties:
            return PolicyVerdict(
                Decision.SUPPRESS, ("user_blocked_payee",), "payee is on the user's blocklist"
            )

        flags: list[str] = []

        # --- amount sanity --------------------------------------------------
        if amount < self.min_amount:
            return PolicyVerdict(Decision.SUPPRESS, ("amount_below_minimum",), "amount below floor")
        if amount > self.max_single:
            return PolicyVerdict(
                Decision.SUPPRESS, ("exceeds_single_transfer_limit",), "above per-transfer limit"
            )

        # --- limits ---------------------------------------------------------
        if ctx.daily_spent + amount > self.daily_limit:
            flags.append("daily_limit_pressure")
        if ctx.monthly_spent + amount > self.monthly_limit:
            flags.append("monthly_limit_pressure")

        # --- funds ----------------------------------------------------------
        headroom = ctx.available_balance - self.balance_buffer
        if headroom > 0 and amount > headroom:
            flags.append("insufficient_balance")
        elif headroom <= 0:
            flags.append("no_estimated_balance")

        # --- scheduling -----------------------------------------------------
        if cand.status is RecurrenceStatus.RECENTLY_PAID:
            flags.append("already_paid_this_cycle")
        if cand.status is RecurrenceStatus.OVERDUE:
            flags.append("overdue_suggested_today")
        if (
            ctx.is_cutoff_passed
            and cand.scheduled_for <= ctx.as_of
            and cand.status in (RecurrenceStatus.DUE_NOW, RecurrenceStatus.OVERDUE)
        ):
            flags.append("after_cutoff_next_day_estimate")

        blocking_flags = {
            "insufficient_balance",
            "no_estimated_balance",
            "daily_limit_pressure",
            "monthly_limit_pressure",
        }
        if blocking_flags & set(flags):
            return PolicyVerdict(
                Decision.SUPPRESS, tuple(flags), "insufficient headroom or limit reached"
            )
        return PolicyVerdict(Decision.ALLOW, tuple(flags))

    def batch(
        self, candidates, ctx: PolicyContext
    ) -> list[tuple[TransferCandidate, PolicyVerdict]]:
        return [(c, self.evaluate(c, ctx)) for c in candidates]


__all__ = ["Decision", "PolicyContext", "PolicyEngine", "PolicyVerdict"]
