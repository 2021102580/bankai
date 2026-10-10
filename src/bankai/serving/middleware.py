
from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from bankai.logging_setup import bind_request_id, bind_user_id, get_logger
from bankai.serving import metrics

log = get_logger("bankai.access")

HTTP_422_UNPROCESSABLE = 422

REQUEST_ID_HEADER = "X-Request-ID"
_MAX_ID_LEN = 64


def _is_safe_token(value: str | None) -> bool:
    return bool(
        value and 0 < len(value) <= _MAX_ID_LEN and all(c.isalnum() or c in "-_." for c in value)
    )


def _endpoint_label(request: Request) -> str:
    route = request.scope.get("route")
    path = getattr(route, "path", None) or request.url.path
    parts = []
    for part in path.split("/"):
        if part.startswith("{") or (part.isdigit() and len(part) > 4):
            parts.append("{}")
        else:
            parts.append(part)
    return "/".join(parts)


class RequestContextMiddleware(BaseHTTPMiddleware):

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        incoming = request.headers.get(REQUEST_ID_HEADER)
        request_id = incoming if _is_safe_token(incoming) else uuid.uuid4().hex
        bind_request_id(request_id)
        bind_user_id(None)
        request.state.request_id = request_id
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            log.exception(
                "request failed",
                extra={
                    "path": request.url.path,
                    "method": request.method,
                    "endpoint": _endpoint_label(request),
                },
            )
            raise
        latency_ms = (time.perf_counter() - started) * 1000.0
        response.headers[REQUEST_ID_HEADER] = request_id
        log.info(
            "request",
            extra={
                "path": request.url.path,
                "method": request.method,
                "status": response.status_code,
                "latency_ms": round(latency_ms, 2),
                "endpoint": _endpoint_label(request),
            },
        )
        return response


class MetricsMiddleware(BaseHTTPMiddleware):

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        metrics.INFLIGHT.inc()
        endpoint = _endpoint_label(request)
        started = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            metrics.INFLIGHT.dec()
            metrics.REQUESTS.labels(
                endpoint=endpoint, method=request.method, status=str(status_code)
            ).inc()
            metrics.REQUEST_LATENCY.labels(endpoint=endpoint, method=request.method).observe(
                time.perf_counter() - started
            )


class ErrorHandlingMiddleware(BaseHTTPMiddleware):

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        try:
            return await call_next(request)
        except Exception as exc:  # pragma: no cover - last line of defence
            log.exception("unhandled error", extra={"path": request.url.path})
            return error_response(
                500,
                "internal_error",
                "An unexpected error occurred",
                request,
                details={"type": type(exc).__name__},
            )


# ---------------------------------------------------------------- responses
def error_response(
    status_code: int,
    error: str,
    message: str,
    request: Request,
    details: dict | None = None,
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    payload: dict = {"error": error, "message": message, "request_id": request_id}
    if details:
        payload["details"] = details
    return JSONResponse(
        status_code=status_code,
        content=payload,
        headers={REQUEST_ID_HEADER: request_id or ""} if request_id else None,
    )


def _safe_validation_details(exc: RequestValidationError) -> list[dict]:
    out = []
    for err in exc.errors()[:10]:
        out.append(
            {
                "field": ".".join(str(p) for p in err.get("loc", ()) if p != "body"),
                "type": err.get("type"),
                "message": err.get("msg"),
            }
        )
    return out


# ------------------------------------------------------------ app handlers
def register_exception_handlers(app: FastAPI) -> None:

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        return error_response(
            HTTP_422_UNPROCESSABLE,
            "validation_error",
            "Request payload failed validation",
            request,
            details={"errors": _safe_validation_details(exc)},
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {
            404: "not_found",
            401: "unauthorized",
            403: "forbidden",
            405: "method_not_allowed",
            429: "rate_limited",
        }.get(exc.status_code, "http_error")
        return error_response(exc.status_code, code, str(exc.detail), request)

    @app.exception_handler(ValueError)
    async def _value(request: Request, exc: ValueError) -> JSONResponse:
        return error_response(400, "bad_request", str(exc), request)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:  # pragma: no cover
        log.exception("unhandled error", extra={"path": request.url.path})
        return error_response(
            500,
            "internal_error",
            "An unexpected error occurred",
            request,
            details={"type": type(exc).__name__},
        )


def install_middleware(app: FastAPI) -> None:
    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(MetricsMiddleware)
    app.add_middleware(RequestContextMiddleware)
    register_exception_handlers(app)


__all__ = [
    "REQUEST_ID_HEADER",
    "ErrorHandlingMiddleware",
    "MetricsMiddleware",
    "RequestContextMiddleware",
    "error_response",
    "install_middleware",
    "register_exception_handlers",
]
