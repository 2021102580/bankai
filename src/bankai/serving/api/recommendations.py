
from __future__ import annotations

import datetime as dt
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status

from bankai.domain.entities import RecurrenceProfile
from bankai.logging_setup import bind_user_id, get_logger
from bankai.ranking.explain import cadence_label
from bankai.ranking.service import RankingResult, RankingService, RequestContext
from bankai.schemas.api import (
    CommitmentsResponse,
    ErrorResponse,
    FeedbackRequest,
    FeedbackResponse,
    Money,
    RecommendationItem,
    RecommendationRequest,
    RecommendationResponse,
    RecurrencePayload,
    RecurringCommitment,
)
from bankai.serving import metrics

log = get_logger(__name__)

router = APIRouter(prefix="/v1", tags=["recommendations"])

ERROR_RESPONSES = {
    404: {"model": ErrorResponse, "description": "User unknown"},
    422: {"model": ErrorResponse, "description": "Request failed validation"},
    500: {"model": ErrorResponse, "description": "Internal error"},
}


def get_service(request: Request) -> RankingService:
    return request.app.state.service


def get_settings_dep(request: Request):
    return request.app.state.settings


ServiceDep = Depends(get_service)


def _as_of_date(request: RecommendationRequest, service: RankingService, settings) -> dt.date:
    if request.as_of is not None:
        return request.as_of.date()
    snapshot = service.repo.latest_date() if hasattr(service.repo, "latest_date") else None
    if snapshot is not None:
        return snapshot
    return dt.datetime.now(tz=dt.UTC).date()


def _recurrence_payload(profile: RecurrenceProfile | None) -> RecurrencePayload | None:
    if profile is None:
        return None
    return RecurrencePayload(
        period_days=profile.period_days,
        cadence=cadence_label(profile.period_days),
        confidence=round(profile.confidence, 4),
        amount_stability=round(profile.amount_stability, 4),
        adherence=round(profile.adherence, 4),
        observations=profile.n_observations,
        last_paid_on=profile.last_paid_on,
        next_due_on=profile.next_due_on,
        days_until_due=profile.days_until_due,
    )


@router.post(
    "/recommendations",
    response_model=RecommendationResponse,
    responses=ERROR_RESPONSES,
    summary="Rank the transfer / bill-payment options for a user",
)
def create_recommendations(
    payload: RecommendationRequest,
    response: Response,
    service: RankingService = ServiceDep,
    settings=Depends(get_settings_dep),
) -> RecommendationResponse:
    bind_user_id(payload.user_id)
    horizon = min(payload.horizon_days, settings.max_horizon_days)
    limit = min(payload.limit, settings.max_recommendations)
    as_of = _as_of_date(payload, service, settings)

    ctx = RequestContext(
        user_id=payload.user_id,
        as_of=as_of,
        horizon_days=horizon,
        limit=limit,
        currency=payload.currency,
        excluded=frozenset(payload.excluded_counterparties),
        channel=payload.channel,
        session_intent=payload.session_intent,
        include_recently_paid=payload.include_recently_paid,
    )

    known = _user_exists(service, payload.user_id)
    if not known:
        raise HTTPException(status_code=404, detail=f"unknown user_id: {payload.user_id}")

    result: RankingResult = service.rank(ctx)
    _record_metrics(result)

    request_id = getattr(payload, "_request_id", None) or uuid.uuid4().hex
    debug = None
    if settings.expose_debug_payload and not settings.is_production:
        debug = {
            "latency_ms": round(result.latency_ms, 3),
            "candidates": result.diagnostics.get("n_candidates"),
            "recall_mix": result.diagnostics.get("recall"),
            "scorer": result.diagnostics.get("scorer"),
            "policy_suppressed": result.diagnostics.get("policy_suppressed", 0),
            "n_recurring_detected": result.diagnostics.get("n_recurring"),
            "n_transactions": result.diagnostics.get("n_transactions"),
        }

    response.status_code = status.HTTP_200_OK
    return RecommendationResponse(
        request_id=request_id,
        user_id=payload.user_id,
        model_version=result.model_version,
        generated_at=result.generated_at,
        as_of=as_of,
        horizon_days=horizon,
        count=len(result.recommendations),
        recommendations=[
            RecommendationItem(
                rank=rec.rank,
                counterparty_id=rec.candidate.counterparty_id,
                counterparty_name=rec.candidate.counterparty_name,
                category=rec.candidate.category,
                suggested_amount=Money(
                    value=rec.candidate.suggested_amount, currency=rec.candidate.currency
                ),
                amount_range=(
                    Money(
                        value=round(rec.candidate.recurrence.amount_low, 2),
                        currency=rec.candidate.currency,
                    )
                    if rec.candidate.recurrence and rec.candidate.recurrence.amount_low is not None
                    else None
                ),
                scheduled_for=rec.candidate.scheduled_for,
                status=rec.candidate.status,
                score=rec.score,
                confidence=rec.confidence,
                recurrence=_recurrence_payload(rec.candidate.recurrence),
                explanations=[
                    {"code": e.code, "text": e.text, "weight": round(e.weight, 4)}
                    for e in rec.explanations
                ],
                policy_flags=rec.policy_flags,
                source=rec.candidate.source,
            )
            for rec in result.recommendations
        ],
        degraded=result.degraded,
        degraded_reason=result.degradation_reason,
        debug=debug,
    )


def _user_exists(service: RankingService, user_id: str) -> bool:
    users = service.repo.users()
    return user_id in set(users)


def _record_metrics(result: RankingResult) -> None:
    n = len(result.recommendations)
    metrics.RECOMMENDATIONS_SERVED.labels(
        bucket=metrics.count_bucket(n),
        status="ok" if not result.degraded else "degraded",
        degraded=str(result.degraded).lower(),
    ).inc()
    pool = result.diagnostics.get("n_candidates")
    if isinstance(pool, int):
        metrics.CANDIDATES_GENERATED.observe(pool)
    for rec in result.recommendations:
        metrics.RANK_SCORE.observe(rec.score)
        metrics.RECURRENCE_STATUS.labels(status=rec.candidate.status.value).inc()
    for flag in ("policy_suppressed",):
        if result.diagnostics.get(flag):
            metrics.POLICY_SUPPRESSED.labels(reason="aggregate").inc(
                result.diagnostics[flag]  # type: ignore[arg-type]
            )


@router.get(
    "/users/{user_id}/commitments",
    response_model=CommitmentsResponse,
    responses=ERROR_RESPONSES,
    summary="Detected standing commitments (recurring payments) for a user",
)
def get_commitments(
    user_id: str,
    as_of: dt.date | None = Query(default=None, description="Point-in-time evaluation date"),
    service: RankingService = ServiceDep,
) -> CommitmentsResponse:
    bind_user_id(user_id)
    if not _user_exists(service, user_id):
        raise HTTPException(status_code=404, detail=f"unknown user_id: {user_id}")
    rows = service.recurring_summary(user_id, as_of)
    if not rows:
        return CommitmentsResponse(
            user_id=user_id,
            as_of=as_of or dt.date.today(),
            count=0,
            total_monthly_commitment=0.0,
            commitments=[],
        )

    monthly = 0.0
    commitments: list[RecurringCommitment] = []
    for row in rows:
        period = row.get("period_days") or 0.0
        amount = row.get("expected_amount") or 0.0
        factor = 30.436875 / period if period > 0 else 0.0
        monthly += amount * factor
        commitments.append(
            RecurringCommitment(
                counterparty_id=row["counterparty_id"],
                counterparty_name=row["counterparty_name"],
                category=row["category"],
                period_days=row.get("period_days"),
                confidence=row["confidence"],
                adherence=row["adherence"],
                expected_amount=row.get("expected_amount"),
                last_paid_on=row.get("last_paid_on"),
                next_due_on=row.get("next_due_on"),
                status=row["status"],
                observations=row["observations"],
            )
        )
    return CommitmentsResponse(
        user_id=user_id,
        as_of=as_of
        or max((c.last_paid_on for c in commitments if c.last_paid_on), default=dt.date.today()),
        count=len(commitments),
        total_monthly_commitment=round(monthly, 2),
        commitments=commitments,
    )


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Record what the user actually did with a suggestion (training signal)",
)
def post_feedback(
    payload: FeedbackRequest, service: RankingService = ServiceDep
) -> FeedbackResponse:
    bind_user_id(payload.user_id)
    decision_id = None
    try:
        decision_id = uuid.uuid4().hex
        service.repo.record_decision(
            decision_id=decision_id,
            user_id=payload.user_id,
            decided_at=payload.occurred_at or dt.datetime.now(tz=dt.UTC),
            counterparty_id=payload.counterparty_id,
            amount=(payload.amount.value if payload.amount else 0.0),
            source=payload.action,
            served_rank=payload.served_rank,
            was_executed=payload.action == "executed",
        )
    except Exception:
        log.exception("failed to record feedback", extra={"user_id": payload.user_id})
    return FeedbackResponse(accepted=True, request_id=uuid.uuid4().hex, decision_id=decision_id)
