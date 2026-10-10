

from __future__ import annotations

from fastapi import APIRouter, Request, Response

from bankai.logging_setup import get_logger
from bankai.schemas.api import HealthResponse
from bankai.serving.metrics import CONTENT_TYPE, render

log = get_logger(__name__)

router = APIRouter(tags=["ops"])
SERVICE_NAME = "bankai-recommender"


@router.get("/healthz", response_model=HealthResponse, summary="Liveness probe")
def healthz(request: Request) -> HealthResponse:
    return HealthResponse(
        status="ok",
        service=SERVICE_NAME,
        version=str(request.app.version),
        model_version=getattr(request.app.state.registry, "_loaded", None)
        and request.app.state.registry._loaded.version,
        checks={"uptime": True},
    )


@router.get("/readyz", response_model=HealthResponse, summary="Readiness probe")
def readyz(request: Request, response: Response) -> HealthResponse:
    checks: dict[str, object] = {}
    healthy = True

    registry = request.app.state.registry
    try:
        loaded = registry.get()
        checks["model"] = {
            "version": loaded.version,
            "healthy": loaded.healthy,
            "reason": loaded.degradation_reason,
            "trained_until": str(loaded.manifest.trained_until) if loaded.healthy else None,
        }
        if not loaded.healthy:
            healthy = False
    except Exception as exc:
        checks["model"] = {"error": type(exc).__name__}
        healthy = False

    try:
        repository = request.app.state.repository
        n = len(repository.users())
        checks["data_source"] = {"status": "ok", "users": n}
        if n == 0:
            checks["data_source"] = {"status": "empty"}
            healthy = False
    except Exception as exc:
        checks["data_source"] = {"error": type(exc).__name__, "status": "unavailable"}
        healthy = False

    status_value = "ok" if healthy else "unhealthy"
    if not response.headers.get("X-Request-ID"):
        response.status_code = 200 if healthy else 503
    return HealthResponse(
        status=status_value,
        service=SERVICE_NAME,
        version=str(request.app.version),
        model_version=checks.get("model", {}).get("version")
        if isinstance(checks.get("model"), dict)
        else None,
        checks=checks,
    )


@router.get("/metrics", include_in_schema=False, summary="Prometheus metrics")
def prometheus_metrics() -> Response:
    return Response(content=render(), media_type=CONTENT_TYPE)


@router.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    return {
        "service": SERVICE_NAME,
        "docs": "/docs",
        "endpoints": ["/v1/recommendations", "/v1/users/{user_id}/commitments", "/v1/feedback"],
    }


__all__ = ["healthz", "readyz", "router"]
