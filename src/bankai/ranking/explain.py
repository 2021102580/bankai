

from __future__ import annotations

import datetime as dt

from bankai.domain.entities import (
    Explanation,
    RecurrenceProfile,
    RecurrenceStatus,
    TransferCandidate,
)
from bankai.features.calendar import classify_billing_day

CADENCE_LABELS: dict[tuple[float, float], str] = {
    (0.0, 1.5): "every day",
    (1.5, 3.5): "every few days",
    (3.5, 9.5): "every week",
    (9.5, 17.0): "every two weeks",
    (17.0, 45.0): "every month",
    (45.0, 75.0): "every two months",
    (75.0, 120.0): "every quarter",
    (120.0, 200.0): "every six months",
    (200.0, 320.0): "yearly",
    (320.0, 1e9): "every few years",
}


def cadence_label(period_days: float | None) -> str:
    if not period_days or period_days <= 0:
        return "occasionally"
    for (lo, hi), label in CADENCE_LABELS.items():
        if lo <= period_days < hi:
            return label
    return "occasionally"


def _money(amount: float, currency: str = "EUR") -> str:
    symbol = {"EUR": "€", "USD": "$", "GBP": "£"}.get(currency, "")
    if symbol:
        return f"{symbol}{amount:,.2f}"
    return f"{amount:,.2f} {currency}"


def explain(
    candidate: TransferCandidate,
    *,
    contributions: dict[str, float] | None = None,
    as_of: dt.date,
    top_n: int = 3,
) -> list[Explanation]:
    rec: RecurrenceProfile | None = candidate.recurrence
    out: list[Explanation] = []
    status = candidate.status

    if status is RecurrenceStatus.OVERDUE and rec and rec.days_until_due is not None:
        late = abs(rec.days_until_due)
        out.append(
            Explanation(
                code="overdue_payment",
                text=(
                    f"{cadence_label(rec.period_days).capitalize()} payment to "
                    f"{candidate.counterparty_name} was expected on "
                    f"{rec.next_due_on:%d %b} — {late} day"
                    f"{'s' if late != 1 else ''} ago."
                ),
                weight=1.00,
            )
        )
    elif status is RecurrenceStatus.DUE_NOW and rec and rec.days_until_due is not None:
        cadence = cadence_label(rec.period_days)
        if rec.days_until_due < 0:
            text = f"{candidate.counterparty_name} is due now — typically {cadence}."
        else:
            plural = "s" if rec.days_until_due != 1 else ""
            text = (
                f"{candidate.counterparty_name} is due in {rec.days_until_due} day{plural}, "
                f"as it is every {cadence.removeprefix('every ')}."
            )
        out.append(Explanation(code="due_now", text=text, weight=0.95))
    elif status is RecurrenceStatus.DUE_SOON and rec and rec.days_until_due is not None:
        out.append(
            Explanation(
                code="due_soon",
                text=(
                    f"{candidate.counterparty_name} is usually paid on day "
                    f"{rec.next_due_on.day if rec.next_due_on else '?'} of the month — "
                    f"next expected {candidate.scheduled_for:%d %b}."
                ),
                weight=0.85,
            )
        )
    elif status is RecurrenceStatus.RECENTLY_PAID and rec and rec.last_paid_on:
        out.append(
            Explanation(
                code="already_paid",
                text=(
                    f"You paid {candidate.counterparty_name} on {rec.last_paid_on:%d %b} — "
                    f"the next one is expected {rec.next_due_on:%d %b}."
                )
                if rec.next_due_on
                else f"You paid {candidate.counterparty_name} on {rec.last_paid_on:%d %b}.",
                weight=0.60,
            )
        )
    elif status is RecurrenceStatus.RECURRING and rec and rec.next_due_on:
        out.append(
            Explanation(
                code="upcoming_schedule",
                text=(
                    f"{cadence_label(rec.period_days).capitalize()} payment to "
                    f"{candidate.counterparty_name}; next expected {rec.next_due_on:%d %b}."
                ),
                weight=0.50,
            )
        )

    # --- pattern evidence --------------------------------------------------
    if rec and rec.is_recurring and rec.n_observations >= 3:
        stability_pct = round(rec.amount_stability * 100)
        out.append(
            Explanation(
                code="regular_pattern",
                text=(
                    f"You have paid {candidate.counterparty_name} {rec.n_observations} times "
                    f"{cadence_label(rec.period_days)}, usually for about "
                    f"{_money(candidate.suggested_amount, candidate.currency)} "
                    f"(amount consistent {stability_pct}% of the time)."
                ),
                weight=0.70,
            )
        )
        if (
            rec.days_until_due is not None
            and rec.days_until_due > 0
            and status
            in (
                RecurrenceStatus.DUE_SOON,
                RecurrenceStatus.RECURRING,
            )
        ):
            out.append(
                Explanation(
                    code="timing_match",
                    text=(
                        f"Payments to this payee land around the "
                        f"{classify_billing_day(rec.next_due_on) if rec.next_due_on else 'same'} "
                        f"part of the month, matching your history."
                    ),
                    weight=0.45,
                )
            )
    elif not rec or rec.n_observations <= 2:
        out.append(
            Explanation(
                code="recent_activity",
                text=f"Based on your recent payments to {candidate.counterparty_name}.",
                weight=0.30,
            )
        )

    # --- amount confidence -------------------------------------------------
    if rec and rec.amount_low is not None and rec.amount_high is not None:
        if (rec.amount_high - rec.amount_low) / max(candidate.suggested_amount, 1.0) < 0.15:
            out.append(
                Explanation(
                    code="amount_confidence",
                    text="The amount is a fixed-price payment, so it will not vary much.",
                    weight=0.35,
                )
            )
        else:
            out.append(
                Explanation(
                    code="amount_estimate",
                    text=(
                        f"Amounts vary; recent payments were between "
                        f"{_money(rec.amount_low, candidate.currency)} and "
                        f"{_money(rec.amount_high, candidate.currency)}."
                    ),
                    weight=0.35,
                )
            )

    # --- model evidence ----------------------------------------------------
    if contributions:
        drivers = sorted(contributions.items(), key=lambda kv: -abs(kv[1]))
        seen: set[str] = set()
        for name, value in drivers:
            text = _CONTRIBUTION_TEXT.get(name)
            if text is None or name in seen or abs(value) < 0.05:
                continue
            seen.add(name)
            out.append(
                Explanation(
                    code=f"model:{name}",
                    text=text,
                    weight=float(min(1.0, abs(value))),
                )
            )
            if len(seen) >= top_n:
                break

    out.sort(key=lambda e: -e.weight)
    return out[: top_n + 2]


_CONTRIBUTION_TEXT: dict[str, str] = {
    # --- due-date urgency ---------------------------------------------------
    "overdue": "This payment is already past its usual date.",
    "due_now": "This payment is due right now.",
    "due_soon": "This payment falls due within the window you asked for.",
    "due_horizon_decay": "The usual payment date is close to today.",
    "recurring": "You pay this payee on a regular schedule.",
    "recurring_periodicity": "This payee follows a recognised billing cycle.",
    "horizon_proximity": "The next payment falls inside the window you asked for.",
    "days_until_due": "The next expected payment date is close.",
    "days_until_due_abs": "The gap to the usual payment date is small.",
    "already_paid_this_cycle": "You have already paid this in the current cycle.",
    # --- schedule quality ---------------------------------------------------
    "recurrence_confidence": "The repeating pattern here is very clear in your history.",
    "schedule_stability": "The timing of these payments is unusually consistent.",
    "periodicity_confidence": "The interval between these payments barely changes.",
    "recurrence_regularity": "These payments arrive at a steady interval.",
    "due_weekday_match": "You usually pay this on the same day of the week.",
    "due_is_mid_month": "You usually pay this around the middle of the month.",
    "due_is_month_boundary": "You usually pay this at the start or end of the month.",
    "typical_weekday": "You have a habit of paying this on the same weekday.",
    # --- consistency and history -------------------------------------------
    "adherence": "You have honoured this schedule reliably.",
    "observations_log": "You have a long history with this payee.",
    "n_observations": "You have paid this payee many times before.",
    "days_since_first_log": "This is a long-standing payment, not a one-off.",
    "days_since_last": "You used this payee not long ago.",
    "recency_ratio": "You paid this payee recently, which usually means it is due again.",
    "recency_rank_pct": "This is one of your most recently used payees.",
    "count_90d": "You have made several payments here in the last three months.",
    "count_365d": "You have used this payee repeatedly over the past year.",
    "active_month_ratio": "You use this payee in most months.",
    "distinct_months": "This payee has been paid in many different months.",
    "is_new_this_month": "This is a payment you only started making this month.",
    # --- amount -------------------------------------------------------------
    "amount_is_fixed": "The amount is almost always the same.",
    "amount_stability": "The amount barely varies between payments.",
    "amount_dispersion": "The amount varies quite a bit from payment to payment.",
    "amount_matches_last": "This matches what you paid last time.",
    "expected_amount": "The amount comes from your usual payment value.",
    "amount_vs_user_median": "This is larger than your typical payment.",
    "share_of_monthly_spend": "This is a large share of your usual monthly spend.",
    "spend_share": "This payee absorbs a large share of your spending.",
    "spend_90d": "You have committed a notable amount to this payee recently.",
    "spend_90d_log": "This is a meaningful commitment for your account.",
    "amount_log": "The payment value is substantial for your account.",
    "expected_amount_log": "The payment value is in line with your normal level.",
    # --- affinity and similarity -------------------------------------------
    "tx_share": "This is one of your most frequently used payees.",
    "category_share": "You spend a lot in this category.",
    "category_affinity": "This category is a consistent part of your spending.",
    "is_dominant_category": "This is your main spending category.",
    "text_similarity": "This payee is much like the ones you use most.",
    "latent_similarity": "Customers with your payment habits also use this payee.",
    "similarity": "This payee behaves like the ones you already use.",
    "novelty": "This is a new or unusual payee for you.",
    "is_known_payee": "You have paid this payee before.",
    "channel_diversity": "You reach this payee through several channels.",
    # --- negative / risk ----------------------------------------------------
    "missed_cycles": "You may have skipped a recent cycle of this payment.",
    "is_recently_paid": "You paid this very recently, so the next one is not due yet.",
    "success_rate": "Payments to this payee have gone through reliably.",
    "failed_recent": "A recent payment to this payee failed.",
    "overdue_suggested_today": "This payment is late, so it is proposed for today.",
    "after_cutoff_next_day_estimate": "It is past today's cut-off, so this would settle tomorrow.",
    # --- cadence ------------------------------------------------------------
    "period_days": "This payee follows a fixed billing cycle.",
    "period_log": "The interval between payments is well established.",
    "period_is_monthly": "This is a monthly payment.",
    "period_is_weekly": "This is a weekly payment.",
    "period_is_quarterly": "This is a quarterly payment.",
}


__all__ = ["Explanation", "cadence_label", "explain"]
