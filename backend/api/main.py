import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.core.cache import redis_client
from backend.core.config import settings
from backend.core.error_handler import register_exception_handlers
from backend.core.log_redaction import redact_url
from backend.core.logging import setup_logging
from backend.core.storage import object_storage
from backend.db.session import SessionLocal, engine
from backend.modules.ai.providers import close_ai_provider_http_clients
from backend.modules.memory.infrastructure.memory_config import validate_memory_config
from backend.modules.platform.service import PlatformService
from backend.modules.rag.infrastructure.rag_config import validate_rag_config
from backend.observability import setup_observability
from backend.observability.service import close_observability_http_client
from backend.workers.async_dispatch import log_eager_mode_startup_warning

from .middleware.correlation_id import CorrelationIdMiddleware
from .middleware.csrf import CSRFMiddleware
from .middleware.public_rate_limit import PublicRateLimitMiddleware
from .middleware.request_logging import RequestLoggingMiddleware
from .middleware.security_headers import SecurityHeadersMiddleware
from .router import api_router
from .v1.health import health_router

setup_logging()

logger = logging.getLogger("backend.startup")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "Application startup app=%s env=%s log_level=%s",
        settings.APP_NAME,
        settings.APP_ENV,
        settings.LOG_LEVEL.upper(),
    )
    log_eager_mode_startup_warning()
    validate_rag_config()
    validate_memory_config()
    await object_storage.ensure_bucket()
    try:
        await redis_client.ping()
        logger.info("Redis connection established url=%s", redact_url(settings.REDIS_URL))
    except Exception:
        logger.warning("Redis ping failed during startup", exc_info=True)
    async with SessionLocal() as db:
        platform_service = PlatformService(db)
        await platform_service.ensure_defaults()
        logger.info("Platform defaults ensured")
    logger.info("Application startup complete")
    yield
    logger.info("Application shutdown started")
    await close_ai_provider_http_clients()
    await close_observability_http_client()
    await redis_client.aclose()
    await engine.dispose()
    logger.info("Application shutdown complete")


app = FastAPI(
    title=settings.APP_NAME,
    docs_url="/docs" if settings.APP_ENV != "production" else None,
    redoc_url="/redoc" if settings.APP_ENV != "production" else None,
    lifespan=lifespan,
)

app.add_middleware(CorrelationIdMiddleware)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(PublicRateLimitMiddleware)
app.add_middleware(CSRFMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-CSRF-Token"],
)

register_exception_handlers(app)
app.include_router(api_router)
app.include_router(health_router)

# Metrics and OTLP instrumentation must register before the ASGI app starts.
setup_observability(app)
