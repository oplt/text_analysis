from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response

from backend.core.config import settings

SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
CSRF_EXEMPT_PATHS = {
    "/api/v1/auth/sign-in",
    "/api/v1/auth/sign-up",
    "/api/v1/auth/forgot-password",
    "/api/v1/auth/reset-password",
    "/api/v1/auth/verify-email",
    "/api/v1/auth/resend-verification",
    "/health/live",
}


class CSRFMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        if request.method in SAFE_METHODS or request.url.path in CSRF_EXEMPT_PATHS:
            return await call_next(request)

        csrf_cookie = request.cookies.get(settings.CSRF_COOKIE_NAME)
        csrf_header = request.headers.get(settings.CSRF_HEADER_NAME)
        auth_cookie = request.cookies.get(settings.ACCESS_COOKIE_NAME) or request.cookies.get(
            settings.REFRESH_COOKIE_NAME
        )
        if auth_cookie and (not csrf_cookie or not csrf_header or csrf_cookie != csrf_header):
            return JSONResponse(status_code=403, content={"detail": "CSRF validation failed"})

        return await call_next(request)
