

from __future__ import annotations

import datetime as dt
import json
import sqlite3
import threading
from collections import OrderedDict
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

from bankai.domain.entities import (
    Channel,
    Direction,
    Transaction,
    TransactionStatus,
)
from bankai.logging_setup import get_logger

log = get_logger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id      TEXT PRIMARY KEY,
    user_id             TEXT NOT NULL,
    posted_at           TEXT NOT NULL,
    value_date          TEXT NOT NULL,
    amount              REAL NOT NULL,
    currency            TEXT NOT NULL,
    direction           TEXT NOT NULL,
    counterparty_id     TEXT NOT NULL,
    counterparty_name   TEXT NOT NULL,
    counterparty_category TEXT NOT NULL DEFAULT 'uncategorised',
    description         TEXT,
    channel             TEXT NOT NULL DEFAULT 'mobile_app',
    status              TEXT NOT NULL DEFAULT 'posted',
    balance_after       REAL,
    raw_category        TEXT
);
CREATE INDEX IF NOT EXISTS ix_tx_user_date ON transactions(user_id, value_date);
CREATE INDEX IF NOT EXISTS ix_tx_user_cp   ON transactions(user_id, counterparty_id, value_date);

CREATE TABLE IF NOT EXISTS user_accounts (
    user_id           TEXT PRIMARY KEY,
    -- NULL means "balance not observable from the account service"; callers
    -- fall back to the ledger's point-in-time balance rather than assuming zero.
    available_balance REAL,
    currency          TEXT NOT NULL DEFAULT 'EUR',
    daily_limit       REAL,
    monthly_limit     REAL,
    blocked_counterparties TEXT NOT NULL DEFAULT '[]',
    updated_at        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS counterparty_directory (
    counterparty_id   TEXT PRIMARY KEY,
    display_name      TEXT NOT NULL,
    category          TEXT NOT NULL,
    country           TEXT,
    is_sanctioned     INTEGER NOT NULL DEFAULT 0,
    requires_memo     INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS transfer_decisions (
    decision_id    TEXT PRIMARY KEY,
    user_id        TEXT NOT NULL,
    decided_at     TEXT NOT NULL,
    counterparty_id TEXT NOT NULL,
    amount         REAL NOT NULL,
    source         TEXT NOT NULL DEFAULT 'unknown',
    served_rank    INTEGER,
    was_executed   INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS ix_dec_user ON transfer_decisions(user_id, decided_at);
"""


class TransactionRepository(Protocol):
    def transactions_for_user(
        self, user_id: str, *, since: dt.date | None = None
    ) -> list[Transaction]: ...

    def users(self) -> list[str]: ...

    def all_transactions(self, limit: int | None = None) -> list[Transaction]: ...

    def account(self, user_id: str) -> dict | None: ...

    def record_decision(self, **kwargs) -> None: ...

    def latest_date(self) -> dt.date | None: ...


class SqliteTransactionRepository:

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._write_lock = threading.Lock()
        self._init_schema()

    # ------------------------------------------------------------- plumbing
    @property
    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return conn

    def _init_schema(self) -> None:
        with self._write_lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    # ---------------------------------------------------------------- write
    def bulk_insert(self, transactions: Iterable[Transaction], *, chunk_size: int = 5000) -> int:
        rows = [
            (
                t.transaction_id,
                t.user_id,
                t.posted_at.isoformat(),
                t.value_date.isoformat(),
                float(t.amount),
                t.currency,
                t.direction.value,
                t.counterparty_id,
                t.counterparty_name,
                t.counterparty_category,
                t.description,
                t.channel.value,
                t.status.value,
                t.balance_after,
                t.raw_category,
            )
            for t in transactions
        ]
        if not rows:
            return 0
        with self._write_lock:
            conn = self._conn
            conn.executemany(
                """
                INSERT OR REPLACE INTO transactions VALUES
                (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                rows,
            )
            conn.commit()
        return len(rows)

    def upsert_account(self, user_id: str, **fields) -> None:
        balance = fields.get("available_balance")
        currency = fields.get("currency", "EUR")
        daily = fields.get("daily_limit")
        monthly = fields.get("monthly_limit")
        blocked = json.dumps(fields.get("blocked_counterparties", []))
        with self._write_lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO user_accounts
                (user_id, available_balance, currency, daily_limit, monthly_limit,
                 blocked_counterparties, updated_at)
                VALUES (?,?,?,?,?,?,?)
                """,
                (
                    user_id,
                    None if balance is None else float(balance),
                    currency,
                    daily,
                    monthly,
                    blocked,
                    dt.datetime.now().isoformat(),
                ),
            )
            self._conn.commit()

    def upsert_counterparty(self, cp_id: str, name: str, category: str, **fields) -> None:
        with self._write_lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO counterparty_directory
                (counterparty_id, display_name, category, country, is_sanctioned, requires_memo)
                VALUES (?,?,?,?,?,?)
                """,
                (
                    cp_id,
                    name,
                    category,
                    fields.get("country"),
                    int(bool(fields.get("is_sanctioned", False))),
                    int(bool(fields.get("requires_memo", False))),
                ),
            )
            self._conn.commit()

    def record_decision(
        self,
        *,
        decision_id: str,
        user_id: str,
        decided_at: dt.datetime,
        counterparty_id: str,
        amount: float,
        source: str = "unknown",
        served_rank: int | None = None,
        was_executed: bool = True,
    ) -> None:
        with self._write_lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO transfer_decisions
                (decision_id, user_id, decided_at, counterparty_id, amount,
                 source, served_rank, was_executed)
                VALUES (?,?,?,?,?,?,?,?)
                """,
                (
                    decision_id,
                    user_id,
                    decided_at.isoformat(),
                    counterparty_id,
                    float(amount),
                    source,
                    served_rank,
                    int(was_executed),
                ),
            )
            self._conn.commit()

    # ----------------------------------------------------------------- read
    @staticmethod
    def _row_to_tx(row: sqlite3.Row) -> Transaction:
        return Transaction(
            transaction_id=row["transaction_id"],
            user_id=row["user_id"],
            posted_at=dt.datetime.fromisoformat(row["posted_at"]),
            value_date=dt.date.fromisoformat(row["value_date"]),
            amount=row["amount"],
            currency=row["currency"],
            direction=Direction(row["direction"]),
            counterparty_id=row["counterparty_id"],
            counterparty_name=row["counterparty_name"],
            counterparty_category=row["counterparty_category"],
            description=row["description"],
            channel=Channel(row["channel"]),
            status=TransactionStatus(row["status"]),
            balance_after=row["balance_after"],
            raw_category=row["raw_category"],
        )

    def transactions_for_user(
        self, user_id: str, *, since: dt.date | None = None
    ) -> list[Transaction]:
        if since is not None:
            cur = self._conn.execute(
                "SELECT * FROM transactions WHERE user_id=? AND value_date>=? ORDER BY value_date",
                (user_id, since.isoformat()),
            )
        else:
            cur = self._conn.execute(
                "SELECT * FROM transactions WHERE user_id=? ORDER BY value_date", (user_id,)
            )
        return [self._row_to_tx(r) for r in cur.fetchall()]

    def users(self) -> list[str]:
        cur = self._conn.execute("SELECT DISTINCT user_id FROM transactions ORDER BY user_id")
        return [r[0] for r in cur.fetchall()]

    def all_transactions(self, limit: int | None = None) -> list[Transaction]:
        sql = "SELECT * FROM transactions ORDER BY user_id, value_date"
        if limit:
            sql += f" LIMIT {int(limit)}"
        cur = self._conn.execute(sql)
        return [self._row_to_tx(r) for r in cur.fetchall()]

    def account(self, user_id: str) -> dict | None:
        cur = self._conn.execute("SELECT * FROM user_accounts WHERE user_id=?", (user_id,))
        row = cur.fetchone()
        if row is None:
            return None
        return {
            "user_id": row["user_id"],
            "available_balance": row["available_balance"],
            "currency": row["currency"],
            "daily_limit": row["daily_limit"],
            "monthly_limit": row["monthly_limit"],
            "blocked_counterparties": json.loads(row["blocked_counterparties"] or "[]"),
            "updated_at": row["updated_at"],
        }

    def counterparty(self, counterparty_id: str) -> dict | None:
        cur = self._conn.execute(
            "SELECT * FROM counterparty_directory WHERE counterparty_id=?", (counterparty_id,)
        )
        row = cur.fetchone()
        if row is None:
            return None
        return {
            "counterparty_id": row["counterparty_id"],
            "display_name": row["display_name"],
            "category": row["category"],
            "country": row["country"],
            "is_sanctioned": bool(row["is_sanctioned"]),
            "requires_memo": bool(row["requires_memo"]),
        }

    def latest_date(self) -> dt.date | None:
        cur = self._conn.execute("SELECT MAX(value_date) FROM transactions")
        row = cur.fetchone()
        return dt.date.fromisoformat(row[0]) if row and row[0] else None

    def count(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) FROM transactions")
        return int(cur.fetchone()[0])

    def snapshot_date(self) -> dt.date | None:
        return self.latest_date()


class CachedRepository:
    """TTLed read-through cache in front of any :class:`TransactionRepository`."""

    def __init__(
        self, inner: TransactionRepository, *, ttl_seconds: float = 60.0, max_entries: int = 4096
    ) -> None:
        self.inner = inner
        self.ttl = ttl_seconds
        self.max_entries = max(1, int(max_entries))
        self._cache: OrderedDict[str, tuple[float, list[Transaction]]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def _get(self, key: str) -> list[Transaction] | None:
        import time

        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self.misses += 1
                return None
            stored_at, value = entry
            if time.monotonic() - stored_at > self.ttl:
                del self._cache[key]
                self.misses += 1
                return None
            self._cache.move_to_end(key)
            self.hits += 1
            return value

    def _put(self, key: str, value: list[Transaction]) -> None:
        import time

        with self._lock:
            self._cache[key] = (time.monotonic(), value)
            self._cache.move_to_end(key)
            while len(self._cache) > self.max_entries:
                self._cache.popitem(last=False)

    def invalidate(self, user_id: str | None = None) -> None:
        with self._lock:
            if user_id is None:
                self._cache.clear()
            else:
                for key in [k for k in self._cache if k.startswith(f"{user_id}|")]:
                    del self._cache[key]

    def transactions_for_user(
        self, user_id: str, *, since: dt.date | None = None
    ) -> list[Transaction]:
        key = f"{user_id}|{since.isoformat() if since else 'all'}"
        cached = self._get(key)
        if cached is not None:
            return cached
        value = self.inner.transactions_for_user(user_id, since=since)
        self._put(key, value)
        return value

    def users(self) -> list[str]:
        return self.inner.users()

    def all_transactions(self, limit: int | None = None) -> list[Transaction]:
        return self.inner.all_transactions(limit=limit)

    def account(self, user_id: str) -> dict | None:
        return self.inner.account(user_id)

    def record_decision(self, **kwargs) -> None:
        self.inner.record_decision(**kwargs)
        self.invalidate(kwargs.get("user_id"))

    def latest_date(self) -> dt.date | None:
        return self.inner.latest_date()


class InMemoryRepository:

    def __init__(self, transactions: list[Transaction] | None = None) -> None:
        self._by_user: dict[str, list[Transaction]] = {}
        self._accounts: dict[str, dict] = {}
        for tx in transactions or []:
            self._by_user.setdefault(tx.user_id, []).append(tx)
        for txs in self._by_user.values():
            txs.sort(key=lambda t: (t.value_date, t.posted_at))

    def add(self, transactions: Iterable[Transaction]) -> None:
        for tx in transactions:
            self._by_user.setdefault(tx.user_id, []).append(tx)
        for txs in self._by_user.values():
            txs.sort(key=lambda t: (t.value_date, t.posted_at))

    def transactions_for_user(
        self, user_id: str, *, since: dt.date | None = None
    ) -> list[Transaction]:
        txs = self._by_user.get(user_id, [])
        if since is not None:
            return [t for t in txs if t.value_date >= since]
        return list(txs)

    def users(self) -> list[str]:
        return sorted(self._by_user)

    def all_transactions(self, limit: int | None = None) -> list[Transaction]:
        flat = [t for txs in self._by_user.values() for t in txs]
        flat.sort(key=lambda t: (t.user_id, t.value_date))
        return flat[:limit] if limit else flat

    def account(self, user_id: str) -> dict | None:
        return self._accounts.get(user_id)

    def set_account(self, user_id: str, **fields) -> None:
        self._accounts[user_id] = {
            "user_id": user_id,
            "available_balance": fields.get("available_balance"),
            "currency": fields.get("currency", "EUR"),
            "daily_limit": fields.get("daily_limit"),
            "monthly_limit": fields.get("monthly_limit"),
            "blocked_counterparties": fields.get("blocked_counterparties", []),
            "updated_at": dt.datetime.now().isoformat(),
        }

    def record_decision(self, **kwargs) -> None:
        return None

    def latest_date(self) -> dt.date | None:
        dates = [t.value_date for txs in self._by_user.values() for t in txs]
        return max(dates) if dates else None


__all__ = [
    "CachedRepository",
    "InMemoryRepository",
    "SqliteTransactionRepository",
    "TransactionRepository",
]
