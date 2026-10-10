"""Shared fixtures.

The suite runs against a small, deterministic corpus so that a failure is
reproducible and the whole thing finishes in seconds.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from bankai.config import Settings
from bankai.data.store import InMemoryRepository, SqliteTransactionRepository
from bankai.data.synthetic import generate_dataset
from bankai.domain.entities import (
    Channel,
    Direction,
    Transaction,
    TransactionStatus,
)

AS_OF = dt.date(2024, 6, 30)


# --------------------------------------------------------------------- corpus
@pytest.fixture(scope="session")
def corpus() -> tuple[dict[str, list[Transaction]], dt.date]:
    txs, _blueprints, end = generate_dataset(n_users=12, months=18, as_of=AS_OF, seed=17)
    by_user: dict[str, list[Transaction]] = {}
    for t in txs:
        by_user.setdefault(t.user_id, []).append(t)
    return by_user, end


@pytest.fixture(scope="session")
def flat_transactions(corpus) -> list[Transaction]:
    by_user, _ = corpus
    return [t for txs in by_user.values() for t in txs]


@pytest.fixture
def repo(corpus) -> InMemoryRepository:
    by_user, _ = corpus
    memory = InMemoryRepository()
    for user_id, txs in by_user.items():
        memory.add(txs)
        memory.set_account(user_id, available_balance=None, currency="EUR")
    return memory


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        artifact_dir=tmp_path / "artifacts",
        database_path=tmp_path / "data" / "test.db",
        environment="test",
        log_level="WARNING",
        n_users=12,
        n_months=18,
        seed=17,
        enable_policy=True,
        expose_debug_payload=True,
    )


# ------------------------------------------------------------------- builders
def make_transaction(
    *,
    user_id: str = "u_test",
    days_from: dt.date = AS_OF,
    amount: float = 10.0,
    name: str = "Test Payee",
    category: str = "utilities.internet",
    counterparty_id: str | None = None,
    direction: Direction = Direction.OUTBOUND,
    status: TransactionStatus = TransactionStatus.POSTED,
    balance_after: float | None = None,
    posted_offset_hours: int = 10,
) -> Transaction:
    value_date = days_from
    posted = dt.datetime.combine(value_date, dt.time(0, 0)) + dt.timedelta(
        hours=posted_offset_hours
    )
    return Transaction(
        transaction_id=f"tx_{user_id}_{abs(hash((value_date, name)))}",
        user_id=user_id,
        posted_at=posted,
        value_date=value_date,
        amount=amount,
        currency="EUR",
        direction=direction,
        counterparty_id=counterparty_id or f"cp_{name.lower().replace(' ', '_')}",
        counterparty_name=name,
        counterparty_category=category,
        channel=Channel.MOBILE_APP,
        status=status,
        balance_after=balance_after,
    )


def monthly_series(
    *,
    day: int,
    amount: float,
    months: int = 12,
    end: dt.date = AS_OF,
    start: dt.date | None = None,
    name: str = "FiberNet ISP",
    category: str = "utilities.internet",
    jitter_days: int = 0,
    skip_every: int = 0,
    amount_noise: float = 0.0,
    seed: int = 3,
) -> list[Transaction]:
    import calendar

    rng = __import__("numpy").random.default_rng(seed)
    out: list[Transaction] = []
    year, month = end.year, end.month
    for _ in range(months):
        if skip_every and (_ % skip_every == 0):
            month -= 1
            if month == 0:
                year, month = year - 1, 12
            continue
        d = day
        if jitter_days:
            d = day + int(rng.integers(-jitter_days, jitter_days + 1))
        d = max(1, min(d, calendar.monthrange(year, month)[1]))
        value_date = dt.date(year, month, d)
        if value_date > end or (start is not None and value_date < start):
            month -= 1
            if month == 0:
                year, month = year - 1, 12
            continue
        amt = amount * (1.0 + float(rng.normal(0, amount_noise)))
        out.append(
            make_transaction(
                days_from=value_date, amount=round(amt, 2), name=name, category=category
            )
        )
        month -= 1
        if month == 0:
            year, month = year - 1, 12
    return sorted(out, key=lambda t: t.value_date)


@pytest.fixture
def seeded_sqlite(settings, corpus) -> tuple[SqliteTransactionRepository, Settings]:
    by_user, _ = corpus
    store = SqliteTransactionRepository(settings.db_path)
    store.bulk_insert([t for txs in by_user.values() for t in txs])
    for user_id in by_user:
        store.upsert_account(user_id, available_balance=None, currency="EUR")
    return store, settings
