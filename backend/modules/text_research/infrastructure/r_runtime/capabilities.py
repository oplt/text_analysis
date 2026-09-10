"""R feature availability vs local worker runtime readiness."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

import redis

from backend.core.config import settings

logger = logging.getLogger(__name__)

R_WORKER_CAPABILITIES_KEY = "research:r_worker:capabilities"
R_WORKER_CAPABILITIES_TTL_SECONDS = 90


def r_feature_enabled() -> bool:
    """True when the deployment intends to offer R analyses (API-side flag).

    Does **not** require a local ``Rscript`` binary — that is the dedicated
    ``research_r`` worker's concern.
    """
    return bool(settings.RESEARCH_R_ENABLED)


def _sync_redis() -> redis.Redis:
    return redis.from_url(settings.REDIS_URL, decode_responses=True)


def publish_r_worker_capabilities(*, ready: bool, analyses: list[str] | None = None) -> None:
    """Heartbeat from an R-capable Celery worker into Redis."""
    payload = {
        "ready": ready,
        "implementation": "quanteda",
        "implementation_version": "r-quanteda-1",
        "analyses": list(analyses or []),
        "published_at": datetime.now(UTC).isoformat(),
        "queue": settings.RESEARCH_QUEUE_R,
    }
    try:
        client = _sync_redis()
        client.setex(
            R_WORKER_CAPABILITIES_KEY,
            R_WORKER_CAPABILITIES_TTL_SECONDS,
            json.dumps(payload),
        )
    except Exception:
        logger.debug("Failed to publish R worker capabilities", exc_info=True)


def get_r_worker_capabilities() -> dict[str, Any] | None:
    """Read the latest R-worker heartbeat, if any."""
    try:
        client = _sync_redis()
        raw = client.get(R_WORKER_CAPABILITIES_KEY)
        if not raw:
            return None
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except Exception:
        logger.debug("Failed to read R worker capabilities", exc_info=True)
        return None


def refresh_r_worker_heartbeat_if_local_runtime_ready() -> bool:
    """Publish readiness when this process can actually run Rscript."""
    from backend.modules.text_research.infrastructure.engines.r_engine import RAnalysisEngine
    from backend.modules.text_research.infrastructure.r_runtime.runner import r_runtime_available

    if not r_feature_enabled():
        return False
    ready = r_runtime_available()
    publish_r_worker_capabilities(
        ready=ready,
        analyses=sorted(RAnalysisEngine.supported_analyses) if ready else [],
    )
    return ready
