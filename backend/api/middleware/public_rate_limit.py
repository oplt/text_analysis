import logging

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response

from backend.core.cache import redis_client
from backend.core.config import settings

logger = logging.getLogger("backend.rate_limit")


class PublicRateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        max_requests = settings.effective_public_rate_limit_requests
        if max_requests <= 0:
            return await call_next(request)

        if not request.url.path.startswith("/api/") and not request.url.path.startswith("/health/"):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        key = f"rate_limit:public:{client_ip}"
        try:
            count = await redis_client.incr(key)
            if count == 1:
                await redis_client.expire(key, settings.PUBLIC_RATE_LIMIT_WINDOW_SECONDS)
            if count > max_requests:
                ttl = await redis_client.ttl(key)
                return JSONResponse(
                    status_code=429,
                    content={"detail": f"Too many requests. Try again in {ttl} seconds."},
                    headers={"Retry-After": str(ttl)},
                )
        except Exception:
            # Fail open so Redis outages do not take down auth/platform routes.
            logger.warning("public rate limit skipped (redis unavailable)", exc_info=True)

        return await call_next(request)
