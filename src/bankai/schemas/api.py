
from __future__ import annotations

import datetime as dt
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from bankai.domain.entities import RecurrenceStatus


# --------------------------------------------------------------------- common
class Money(BaseModel):
    model_config = ConfigDict(frozen=True)

    value: float = Field(ge=0, le=1e9, description="Amount in major currency units")
    currency: str = Field(min_length=3, max_length=3, pattern="^[A-Za-z]{3}$")

    @field_validator("currency")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.value:,.2f} {self.currency}"


class ExplanationPayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str
    text: str
    weight: float = 0.0


class RecurrencePayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    period_days: float | None = None
    cadence: str = "irregular"
    confidence: float = 0.0
    amount_stability: float = 0.0
    adherence: float = 0.0
    observations: int = 0
    last_paid_on: dt.date | None = None
    next_due_on: dt.date | None = None
    days_until_due: int | None = None


# ------------------------------------------------------------------ requests
class RecommendationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    user_id: str = Field(min_length=1, max_length=128, examples=["u_000042"])
    as_of: dt.datetime | None = Field(
        default=None,
        description="Point-in-time evaluation instant. Defaults to the latest data snapshot.",
    )
    horizon_days: int = Field(default=14, ge=0, le=60)
    limit: int = Field(default=5, ge=1, le=10)
    currency: str = Field(default="EUR", min_length=3, max_length=3, pattern="^[A-Za-z]{3}$")
    channel: Literal["mobile_app", "web", "open_banking", "branch", "api"] = "mobile_app"
    session_intent: str | None = Field(default=None, max_length=64)
    excluded_counterparties: list[str] = Field(default_factory=list, max_length=50)
    include_recently_paid: bool = True

    @field_validator("currency")
    @classmethod
    def _upper_currency(cls, v: str) -> str:
        return v.upper()

    @field_validator("excluded_counterparties")
    @classmethod
    def _dedupe(cls, v: list[str]) -> list[str]:
        return list(dict.fromkeys(v))

    @model_validator(mode="after")
    def _timezone(self) -> RecommendationRequest:
        if self.as_of is not None and self.as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware (e.g. 2024-06-30T09:00:00Z)")
        return self


class FeedbackRequest(BaseModel):

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    user_id: str = Field(min_length=1, max_length=128)
    counterparty_id: str = Field(min_length=1, max_length=128)
    request_id: str | None = Field(default=None, max_length=64)
    served_rank: int | None = Field(default=None, ge=1, le=100)
    action: Literal["executed", "dismissed", "edited", "snoozed"]
    amount: Money | None = None
    occurred_at: dt.datetime | None = None
    model_version: str | None = Field(default=None, max_length=64)


# ----------------------------------------------------------------- responses
class RecommendationItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    rank: int = Field(
        ge=1,
        description="Final position after diversity re-ranking; this is the order to display.",
    )
    counterparty_id: str
    counterparty_name: str
    category: str
    suggested_amount: Money
    amount_range: Money | None = None
    scheduled_for: dt.date
    status: RecurrenceStatus
    score: float = Field(
        ge=0,
        le=1,
        description=(
            "Relevance score before diversity re-ranking. Rank is authoritative; "
            "scores may appear non-monotonic in rank when the list is diversified."
        ),
    )
    confidence: float = Field(
        ge=0,
        le=1,
        description="Strength of the detected payment pattern behind this suggestion.",
    )
    recurrence: RecurrencePayload | None = None
    explanations: list[ExplanationPayload] = Field(default_factory=list)
    policy_flags: list[str] = Field(default_factory=list)
    source: str = "unknown"


class RecommendationResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    request_id: str
    user_id: str
    model_version: str
    generated_at: dt.datetime
    as_of: dt.date
    horizon_days: int
    count: int = Field(ge=0)
    recommendations: list[RecommendationItem]
    degraded: bool = False
    degraded_reason: str | None = None
    debug: dict[str, Any] | None = None


class RecurringCommitment(BaseModel):
    model_config = ConfigDict(frozen=True)

    counterparty_id: str
    counterparty_name: str
    category: str
    period_days: float | None
    confidence: float
    adherence: float
    expected_amount: float | None
    currency: str = "EUR"
    last_paid_on: dt.date | None
    next_due_on: dt.date | None
    status: RecurrenceStatus
    observations: int


class CommitmentsResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    user_id: str
    as_of: dt.date
    count: int
    total_monthly_commitment: float
    currency: str = "EUR"
    commitments: list[RecurringCommitment]


class HealthResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: Literal["ok", "degraded", "unhealthy"]
    service: str
    version: str
    model_version: str | None
    checks: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    error: str
    message: str
    request_id: str | None = None
    details: dict[str, Any] | None = None


class FeedbackResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    accepted: bool
    request_id: str
    decision_id: str | None = None


__all__ = [
    "CommitmentsResponse",
    "ErrorResponse",
    "FeedbackRequest",
    "FeedbackResponse",
    "HealthResponse",
    "Money",
    "RecommendationItem",
    "RecommendationRequest",
    "RecommendationResponse",
    "RecurrencePayload",
    "RecurringCommitment",
]
