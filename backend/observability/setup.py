from typing import Any

from backend.core.config import settings
from backend.observability.otel import setup_observability as setup_runtime_observability
from backend.observability.sentry import setup_sentry


def setup_observability(app: Any | None = None) -> None:
    setup_sentry(
        settings.SENTRY_DSN,
        settings.APP_ENV,
        settings.SENTRY_TRACES_SAMPLE_RATE,
    )
    setup_runtime_observability(app, service_name=settings.OTEL_SERVICE_NAME)
