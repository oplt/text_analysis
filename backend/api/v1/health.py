from fastapi import APIRouter, HTTPException
from sqlalchemy import text

from backend.core.cache import redis_client
from backend.core.config import settings
from backend.db.session import engine

health_router = APIRouter(prefix="/health", tags=["health"])


@health_router.get("/live")
async def live():
    return {"status": "ok"}


@health_router.get("/ready")
async def ready():
    if not settings.HEALTH_READY_PUBLIC and settings.is_production:
        raise HTTPException(status_code=404, detail="Not found")
    checks: dict[str, str] = {}

    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        checks["db"] = "ok"
    except Exception:
        checks["db"] = "error"

    try:
        await redis_client.ping()
        checks["redis"] = "ok"
    except Exception:
        checks["redis"] = "error"

    if settings.celery_broker_url.startswith(("redis://", "rediss://")):
        checks["queue"] = checks["redis"]
    else:
        checks["queue"] = "unknown"

    all_ok = all(v == "ok" for v in checks.values())
    return {"status": "ok" if all_ok else "degraded", "checks": checks}


@health_router.get("/version")
async def version():
    if not settings.HEALTH_VERSION_PUBLIC and settings.is_production:
        raise HTTPException(status_code=404, detail="Not found")
    return {
        "app": settings.APP_NAME,
        "env": settings.APP_ENV,
        "version": "0.1.0",
        "async_jobs": "celery",
    }
