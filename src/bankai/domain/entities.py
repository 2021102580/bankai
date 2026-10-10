
from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Money = Annotated[float, Field(ge=0, description="Non-negative monetary amount")]


class Direction(StrEnum):
    INBOUND = "inbound"
    OUTBOUND = "outbound"


class TransactionStatus(StrEnum):
    POSTED = "posted"
    PENDING = "pending"
    SCHEDULED = "scheduled"
    FAILED = "failed"
    REVERSED = "reversed"


class Channel(StrEnum):
    MOBILE_APP = "mobile_app"
    WEB = "web"
    OPEN_BANKING = "open_banking"
    BRANCH = "branch"
    SCHEDULED = "scheduled"
    API = "api"


class RecurrenceStatus(StrEnum):

    OVERDUE = "overdue"  # expected date already passed and unpaid
    DUE_NOW = "due_now"  # expected within the current tolerance window
    DUE_SOON = "due_soon"  # expected inside the requested horizon
    RECENTLY_PAID = "recently_paid"  # paid this cycle, next due further out
    RECURRING = "recurring"  # regular but not due within the horizon
    ONE_OFF = "one_off"
    UNKNOWN = "unknown"


class RiskTier(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Transaction(BaseModel):

    model_config = ConfigDict(frozen=True, extra="forbid")

    transaction_id: str
    user_id: str
    posted_at: datetime
    value_date: date
    amount: float = Field(ge=0.0, description="Absolute amount, never signed")
    currency: str = Field(min_length=3, max_length=3)
    direction: Direction
    counterparty_id: str
    counterparty_name: str
    counterparty_category: str = "uncategorised"
    description: str | None = None
    channel: Channel = Channel.MOBILE_APP
    status: TransactionStatus = TransactionStatus.POSTED
    balance_after: float | None = None
    raw_category: str | None = None

    @field_validator("currency")
    @classmethod
    def _upper_currency(cls, v: str) -> str:
        return v.upper()

    @model_validator(mode="after")
    def _check_amount(self) -> Transaction:
        if self.amount == 0.0:
            raise ValueError("transaction amount must be strictly positive")
        return self

    @property
    def balance_delta(self) -> float:
        """Effect of this entry on the account balance (credits positive)."""
        return self.amount if self.direction is Direction.INBOUND else -self.amount


class RecurrenceProfile(BaseModel):

    model_config = ConfigDict(frozen=True, extra="forbid")

    counterparty_id: str
    is_recurring: bool
    period_days: float | None = None
    periodicity_confidence: float = Field(ge=0, le=1)
    amount_stability: float = Field(default=0.0, ge=0, le=1)
    schedule_stability: float = Field(default=0.0, ge=0, le=1)
    regularity: float = Field(default=0.0, ge=0, le=1)
    n_observations: int = 0
    adherence: float = Field(default=0.0, ge=0, le=1, doc="Observed / expected occurrences")
    last_paid_on: date | None = None
    next_due_on: date | None = None
    days_until_due: int | None = None
    missed_cycles: int = 0
    median_amount: float | None = None
    amount_dispersion: float = Field(default=0.0, ge=0)
    expected_amount: float | None = None
    amount_low: float | None = None
    amount_high: float | None = None
    cycle_index: int = 0
    status: RecurrenceStatus = RecurrenceStatus.UNKNOWN
    typical_weekday: int | None = None
    category: str = "uncategorised"
    counterparty_name: str = ""

    @property
    def confidence(self) -> float:
        if self.n_observations < 2:
            return 0.0
        parts = (
            self.periodicity_confidence,
            self.schedule_stability,
            self.amount_stability,
            min(1.0, self.n_observations / 6.0),
        )
        # Geometric mean punishes any single weak dimension.
        geo = 1.0
        for p in parts:
            geo *= max(p, 1e-6)
        return float(geo ** (1.0 / len(parts)))

    @property
    def actionable_confidence(self) -> float:
        return self.confidence if self.is_recurring else 0.0


class TransferCandidate(BaseModel):

    model_config = ConfigDict(frozen=True, extra="allow")

    user_id: str
    counterparty_id: str
    counterparty_name: str
    category: str
    suggested_amount: float
    currency: str
    scheduled_for: date
    status: RecurrenceStatus = RecurrenceStatus.UNKNOWN
    recurrence: RecurrenceProfile | None = None
    source: str = "unknown"
    base_score: float = 0.0
    provenance: dict[str, Any] = Field(default_factory=dict)


class Explanation(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str
    text: str
    weight: float = 0.0


class Recommendation(BaseModel):
    model_config = ConfigDict(frozen=True)

    rank: int = Field(ge=1)
    candidate: TransferCandidate
    score: float
    confidence: float
    contributions: dict[str, float] = Field(default_factory=dict)
    explanations: list[Explanation] = Field(default_factory=list)
    policy_flags: list[str] = Field(default_factory=list)


class ModelManifest(BaseModel):

    model_config = ConfigDict(frozen=True)

    model_name: str
    version: str
    created_at: datetime
    trained_until: date
    n_users: int = 0
    n_decision_points: int = 0
    n_training_rows: int = 0
    metrics: dict[str, float] = Field(default_factory=dict)
    feature_names: list[str] = Field(default_factory=list)
    library_versions: dict[str, str] = Field(default_factory=dict)
    fallback_version: str | None = None

    @property
    def artifact_stem(self) -> str:
        return self.model_name
