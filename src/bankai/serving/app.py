
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from bankai.config import Settings, get_settings
from bankai.logging_setup import configure_logging, get_logger
from bankai.serving.api.health import router as health_router
from bankai.serving.api.recommendations import router as recommendations_router
from bankai.serving.metrics import MODEL_DEGRADED, MODEL_INFO
from bankai.serving.middleware import install_middleware

log = get_logger(__name__)

API_VERSION = "1.0.0"

DESCRIPTION = ""


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_format)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        log.info(
            "service starting",
            extra={
                "service": settings.service_name,
                "env": settings.environment,
                "version": API_VERSION,
            },
        )
        registry = app.state.registry
        loaded = registry.load()
        if loaded.healthy:
            MODEL_INFO.labels(
                version=loaded.version,
                trained_until=str(loaded.manifest.trained_until),
            ).set(1)
            log.info(
                "model loaded",
                extra={
                    "version": loaded.version,
                    "trained_until": str(loaded.manifest.trained_until),
                    "ndcg@5": loaded.manifest.metrics.get("ndcg@5"),
                },
            )
        else:
            MODEL_DEGRADED.labels(reason=loaded.degradation_reason or "unknown").set(1)
            log.warning("serving degraded", extra={"reason": loaded.degradation_reason})
        yield
        log.info("service stopping")

    app = FastAPI(
        title="BankAI Transfer Recommendations",
        description=DESCRIPTION,
        version=API_VERSION,
        lifespan=lifespan,
        docs_url="/docs" if not settings.is_production else None,
        redoc_url=None,
        openapi_url="/openapi.json" if not settings.is_production else "/openapi-internal.json",
    )

    from bankai.data.store import CachedRepository, SqliteTransactionRepository
    from bankai.models.registry import ModelRegistry
    from bankai.ranking.service import RankingService

    store = SqliteTransactionRepository(settings.db_path)
    repository = CachedRepository(store, ttl_seconds=60.0)
    registry = ModelRegistry(settings.artifact_dir)
    service = RankingService(repository=repository, registry=registry, settings=settings)

    app.state.settings = settings
    app.state.store = store
    app.state.repository = repository
    app.state.registry = registry
    app.state.service = service

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "X-Request-ID", "Authorization"],
            expose_headers=["X-Request-ID"],
            max_age=600,
        )

    install_middleware(app)
    app.include_router(health_router)
    app.include_router(recommendations_router)
    return app


app = create_app() if os.getenv("BANKAI_EAGER_APP", "1") == "1" else None

__all__ = ["API_VERSION", "app", "create_app"]
