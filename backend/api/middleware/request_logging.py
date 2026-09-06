import logging
from time import perf_counter

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from backend.core.config import settings

logger = logging.getLogger("backend.request")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        started_at = perf_counter()
        response = await call_next(request)
        duration_ms = (perf_counter() - started_at) * 1000
        correlation_id = getattr(request.state, "correlation_id", "n/a")
        log_args = (
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            correlation_id,
        )

        if duration_ms >= settings.SLOW_REQUEST_MS:
            logger.warning(
                "slow_request method=%s path=%s status=%s duration_ms=%.2f correlation_id=%s",
                *log_args,
            )
        elif response.status_code >= 500:
            logger.error(
                "request_complete method=%s path=%s status=%s duration_ms=%.2f correlation_id=%s",
                *log_args,
            )
        elif response.status_code >= 400:
            logger.warning(
                "request_complete method=%s path=%s status=%s duration_ms=%.2f correlation_id=%s",
                *log_args,
            )
        else:
            logger.info(
                "request_complete method=%s path=%s status=%s duration_ms=%.2f correlation_id=%s",
                *log_args,
            )
        return response
