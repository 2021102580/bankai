
from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass, field

import numpy as np

from bankai.domain.entities import Channel, Direction, Transaction, TransactionStatus

NETWORK_ID = uuid.UUID("6f1a0b3e-2c4d-4a5e-8b9f-0a1b2c3d4e5f")

# ---------------------------------------------------------------- merchant zoo

UTILITIES = [
    ("FiberNet ISP", "utilities.internet", (29.0, 60.0), "monthly", 0),
    ("Nordic Telecom Mobile", "utilities.phone", (18.0, 35.0), "monthly", 2),
    ("VoltGrid Energy", "utilities.electricity", (40.0, 190.0), "monthly", 11),
    ("AquaWorks Water", "utilities.water", (18.0, 65.0), "monthly", 18),
    ("CityHeat District Heating", "utilities.heating", (55.0, 240.0), "quarterly", 5),
    ("HomeShield Security", "utilities.security", (22.0, 44.0), "monthly", 7),
    ("StreamBox TV", "entertainment.tv", (29.0, 55.0), "monthly", 15),
]
SUBSCRIPTIONS = [
    ("StreamFlix", "entertainment.video", (9.99, 17.99), "monthly", 21),
    ("TuneWave Music", "entertainment.audio", (10.99, 14.99), "monthly", 25),
    ("CloudVault Storage", "services.cloud", (2.99, 12.99), "monthly", 27),
    ("FitTrack Club", "health.fitness", (19.0, 49.0), "monthly", 3),
    ("NewsDaily Digital", "news.subscription", (4.5, 12.0), "monthly", 9),
    ("GreenRoute Insurance", "insurance.health", (24.0, 78.0), "quarterly", 13),
]
DISCRETIONARY = [
    ("FareGo Transit", "transport.fare", (2.75, 60.0), "weekly", 0),
    ("QuickMart Grocery", "shopping.grocery", (18.0, 160.0), "weekly", 6),
    ("CafeNine Coffee", "dining.coffee", (2.8, 9.5), "weekly", 2),
    ("FitFuel Gym", "health.fitness", (12.0, 60.0), "monthly", 8),
    ("PetCorner Supplies", "shopping.pets", (10.0, 70.0), "monthly", 16),
    ("BookNest", "shopping.books", (8.0, 45.0), "monthly", 24),
    ("TravelHub Lodging", "travel.lodging", (60.0, 480.0), "irregular", 0),
]
COUNTERPARTIES_PERSONAL = [
    "Marta Kowalski",
    "Jonas Berg",
    "Priya Raman",
    "Tom Adeyemi",
    "Lena Fischer",
    "Carlos Mendes",
    "Anya Sokolova",
    "Diego Alvarez",
    "Nour Haddad",
    "Ken Watanabe",
]
LANDLORDS = ["Helsinki Property Group", "Sunrise Estates Ltd", "NordWohn Immobilien"]


@dataclass
class UserBlueprint:
    user_id: str
    salary: float
    salary_day: int
    rent_amount: float
    rent_day: int
    landlord: str
    obligations: list[tuple[str, str, float, str, int]] = field(default_factory=list)
    discretionary: list[tuple[str, str, float, str, int]] = field(default_factory=list)
    loan_amount: float = 0.0
    loan_day: int = 12
    savings_rate: float = 0.08
    reliability: float = 0.94
    spend_scale: float = 1.0


def _counterparty_id(name: str) -> str:
    return f"cp_{uuid.uuid5(NETWORK_ID, name).hex[:12]}"


class _TxnFactory:
    def __init__(self, rng: np.random.Generator, opening_balance: float = 0.0) -> None:
        self.rng = rng
        self.counter = 0
        self.balance = float(opening_balance)

    def make(
        self,
        *,
        user_id: str,
        value_date: dt.date,
        amount: float,
        name: str,
        category: str,
        direction: Direction = Direction.OUTBOUND,
        channel: Channel = Channel.MOBILE_APP,
        status: TransactionStatus = TransactionStatus.POSTED,
        description: str | None = None,
    ) -> Transaction:
        self.counter += 1
        hour = int(self.rng.integers(7, 22))
        posted = dt.datetime.combine(value_date, dt.time(hour, int(self.rng.integers(0, 60))))
        if status is not TransactionStatus.FAILED:
            self.balance += amount if direction is Direction.INBOUND else -amount
        return Transaction(
            transaction_id=f"tx_{user_id}_{self.counter:07d}",
            user_id=user_id,
            posted_at=posted,
            value_date=value_date,
            amount=round(float(amount), 2),
            currency="EUR",
            direction=direction,
            counterparty_id=_counterparty_id(name),
            counterparty_name=name,
            counterparty_category=category,
            description=description or f"{category.replace('.', ' ').title()} payment",
            channel=channel,
            status=status,
            balance_after=round(self.balance, 2),
        )


def _add_months(value: dt.date, months: int) -> dt.date:
    total = value.month - 1 + months
    year = value.year + total // 12
    month = total % 12 + 1
    import calendar

    return dt.date(year, month, min(value.day, calendar.monthrange(year, month)[1]))


def _clamp_day(year: int, month: int, day: int) -> dt.date:
    import calendar

    return dt.date(year, month, min(day, calendar.monthrange(year, month)[1]))


def build_blueprint(rng: np.random.Generator, user_id: str) -> UserBlueprint:
    salary = float(np.round(rng.uniform(2100, 7200), -1))
    rent = float(np.round(float(rng.uniform(0.22, 0.45)) * salary, -1))
    blueprint = UserBlueprint(
        user_id=user_id,
        salary=salary,
        salary_day=int(rng.integers(24, 29)),
        rent_amount=rent,
        rent_day=int(rng.integers(1, 4)),
        landlord=str(rng.choice(LANDLORDS)),
        reliability=float(np.clip(rng.normal(0.94, 0.05), 0.7, 0.995)),
        savings_rate=float(np.clip(rng.normal(0.08, 0.06), 0.0, 0.25)),
        spend_scale=float(np.clip(rng.normal(1.0, 0.3), 0.4, 2.0)),
    )
    if rng.random() < 0.35:
        blueprint.loan_amount = float(np.round(rng.uniform(150, 950), -1))

    n_obligations = int(rng.integers(2, 6))
    pool = UTILITIES + SUBSCRIPTIONS
    chosen = rng.choice(len(pool), size=min(n_obligations, len(pool)), replace=False)
    for i in chosen:
        name, category, (lo, hi), cadence, day = pool[int(i)]
        amount = float(np.round(rng.uniform(lo, hi) * blueprint.spend_scale, 2))
        blueprint.obligations.append((name, category, amount, cadence, day))

    n_disc = int(rng.integers(2, 5))
    dpool = DISCRETIONARY
    dchosen = rng.choice(len(dpool), size=min(n_disc, len(dpool)), replace=False)
    for i in dchosen:
        name, category, (lo, hi), cadence, day = dpool[int(i)]
        amount = float(np.round(rng.uniform(lo, hi) * blueprint.spend_scale, 2))
        blueprint.discretionary.append((name, category, amount, cadence, day))
    return blueprint


def _emit_cycle(
    factory: _TxnFactory,
    bp: UserBlueprint,
    *,
    name: str,
    category: str,
    amount: float,
    cadence: str,
    target_day: int,
    year: int,
    month: int,
) -> list[Transaction]:
    out: list[Transaction] = []
    if cadence == "monthly":
        months = [(year, month)]
    elif cadence == "quarterly":
        if (month - 1) % 3 != 0:
            return out
        months = [(year, month)]
    else:
        return out

    for y, m in months:
        if factory.rng.random() > bp.reliability:
            continue  # skipped cycle
        jitter_days = int(factory.rng.normal(0, 2.0))
        day = target_day + jitter_days
        try:
            value_date = _clamp_day(y, m, day)
        except ValueError:  # pragma: no cover - defensive
            continue
        amt_jitter = float(factory.rng.normal(1.0, 0.045))
        if category in {"utilities.electricity", "utilities.water", "utilities.heating"}:
            # Utility bills swing with the season.
            season = 1.0 + 0.28 * np.cos(2 * np.pi * (m - 1) / 12.0)
            amt_jitter *= season
        amount_paid = max(0.5, amount * amt_jitter)
        status = TransactionStatus.POSTED
        if factory.rng.random() < 0.012:
            status = TransactionStatus.FAILED
        out.append(
            factory.make(
                user_id=bp.user_id,
                value_date=value_date,
                amount=amount_paid,
                name=name,
                category=category,
                status=status,
            )
        )
    return out


def generate_user_transactions(
    bp: UserBlueprint,
    *,
    start: dt.date,
    end: dt.date,
    rng: np.random.Generator,
) -> list[Transaction]:
    factory = _TxnFactory(rng, opening_balance=bp.salary * 1.9)
    txs: list[Transaction] = []

    # ---- payroll (always reliable) ----------------------------------------
    month_cursor = dt.date(start.year, start.month, 1)
    while month_cursor <= end:
        y, m = month_cursor.year, month_cursor.month
        value_date = _clamp_day(y, m, bp.salary_day)
        if value_date >= start:
            txs.append(
                factory.make(
                    user_id=bp.user_id,
                    value_date=value_date,
                    amount=bp.salary,
                    name=f"{bp.user_id} Payroll",
                    category="income.salary",
                    direction=Direction.INBOUND,
                    channel=Channel.SCHEDULED,
                    description="Monthly salary credit",
                )
            )
        month_cursor = _add_months(month_cursor, 1)

    # ---- rent (high reliability, month boundary) --------------------------
    month_cursor = dt.date(start.year, start.month, 1)
    while month_cursor <= end:
        y, m = month_cursor.year, month_cursor.month
        if rng.random() < bp.reliability * 0.98:
            value_date = _clamp_day(y, m, bp.rent_day)
            txs.append(
                factory.make(
                    user_id=bp.user_id,
                    value_date=value_date,
                    amount=bp.rent_amount,
                    name=bp.landlord,
                    category="housing.rent",
                    channel=Channel.SCHEDULED,
                    description="Rent payment",
                )
            )
        month_cursor = _add_months(month_cursor, 1)

    # ---- loan ------------------------------------------------------------
    if bp.loan_amount > 0:
        month_cursor = dt.date(start.year, start.month, 1)
        while month_cursor <= end:
            y, m = month_cursor.year, month_cursor.month
            if rng.random() < 0.985:
                txs.append(
                    factory.make(
                        user_id=bp.user_id,
                        value_date=_clamp_day(y, m, bp.loan_day),
                        amount=bp.loan_amount,
                        name="Meridian Consumer Credit",
                        category="loans.installment",
                        channel=Channel.SCHEDULED,
                        description="Loan instalment",
                    )
                )
            month_cursor = _add_months(month_cursor, 1)

    # ---- monthly obligations ---------------------------------------------
    month_cursor = dt.date(start.year, start.month, 1)
    while month_cursor <= end:
        for name, category, amount, cadence, day in bp.obligations:
            txs.extend(
                _emit_cycle(
                    factory,
                    bp,
                    name=name,
                    category=category,
                    amount=amount,
                    cadence=cadence,
                    target_day=day,
                    year=month_cursor.year,
                    month=month_cursor.month,
                )
            )
        month_cursor = _add_months(month_cursor, 1)

    # ---- savings sweep ----------------------------------------------------
    month_cursor = dt.date(start.year, start.month, 1)
    while month_cursor <= end:
        if bp.savings_rate > 0.005 and rng.random() < 0.6:
            txs.append(
                factory.make(
                    user_id=bp.user_id,
                    value_date=_clamp_day(month_cursor.year, month_cursor.month, 3),
                    amount=float(bp.salary * bp.savings_rate),
                    name="Vault Savings Account",
                    category="savings.transfer",
                    channel=Channel.SCHEDULED,
                    description="Automatic savings",
                )
            )
        month_cursor = _add_months(month_cursor, 1)

    # ---- discretionary behaviour -----------------------------------------
    for name, category, amount, cadence, day in bp.discretionary:
        if cadence == "weekly":
            current = start + dt.timedelta(days=int(rng.integers(0, 7)))
            while current <= end:
                if rng.random() < 0.55:
                    txs.append(
                        factory.make(
                            user_id=bp.user_id,
                            value_date=current,
                            amount=amount * float(rng.uniform(0.6, 1.5)),
                            name=name,
                            category=category,
                        )
                    )
                current += dt.timedelta(days=7)
        elif cadence == "monthly":
            month_cursor = dt.date(start.year, start.month, 1)
            while month_cursor <= end:
                if rng.random() < 0.62:
                    txs.append(
                        factory.make(
                            user_id=bp.user_id,
                            value_date=_clamp_day(
                                month_cursor.year,
                                month_cursor.month,
                                day + int(rng.integers(-3, 4)),
                            ),
                            amount=amount * float(rng.uniform(0.6, 1.4)),
                            name=name,
                            category=category,
                        )
                    )
                month_cursor = _add_months(month_cursor, 1)
        else:  # irregular / occasional
            for _ in range(int(rng.poisson(0.9))):
                offset = int(rng.integers(0, max(1, (end - start).days)))
                txs.append(
                    factory.make(
                        user_id=bp.user_id,
                        value_date=start + dt.timedelta(days=offset),
                        amount=amount * float(rng.uniform(0.5, 1.8)),
                        name=name,
                        category=category,
                    )
                )

    # ---- occasional person-to-person transfers ---------------------------
    for _ in range(int(rng.poisson(2.2))):
        offset = int(rng.integers(0, max(1, (end - start).days)))
        peer = str(rng.choice(COUNTERPARTIES_PERSONAL))
        txs.append(
            factory.make(
                user_id=bp.user_id,
                value_date=start + dt.timedelta(days=offset),
                amount=float(rng.uniform(15, 320)),
                name=peer,
                category="p2p.transfer",
                description="Peer transfer",
            )
        )

    txs.sort(key=lambda t: (t.value_date, t.posted_at))
    _recompute_balances(txs, opening_balance=bp.salary * 1.9)
    return txs


def _recompute_balances(txs: list[Transaction], *, opening_balance: float) -> None:
    balance = float(opening_balance)
    for tx in txs:
        if tx.status is not TransactionStatus.FAILED:
            balance += tx.balance_delta
        object.__setattr__(tx, "balance_after", round(balance, 2))


def generate_dataset(
    *,
    n_users: int = 400,
    months: int = 18,
    as_of: dt.date | None = None,
    seed: int = 7,
) -> tuple[list[Transaction], list[UserBlueprint], dt.date]:
    rng = np.random.default_rng(seed)
    end = as_of or dt.date(2024, 6, 30)
    start = _add_months(end.replace(day=1), -months) + dt.timedelta(days=1)
    all_txs: list[Transaction] = []
    blueprints: list[UserBlueprint] = []
    for i in range(n_users):
        user_id = f"u_{i:06d}"
        bp = build_blueprint(rng, user_id)
        blueprints.append(bp)
        all_txs.extend(generate_user_transactions(bp, start=start, end=end, rng=rng))
    return all_txs, blueprints, end


__all__ = ["UserBlueprint", "build_blueprint", "generate_dataset", "generate_user_transactions"]
