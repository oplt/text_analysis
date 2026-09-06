import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger("backend.error")


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def request_validation_error_handler(request: Request, exc: RequestValidationError):
        logger.warning(
            "request_validation_failed path=%s correlation_id=%s errors=%s",
            request.url.path,
            getattr(request.state, "correlation_id", None),
            len(exc.errors()),
        )
        return JSONResponse(
            status_code=422,
            content={
                "detail": "Request validation failed",
                "errors": exc.errors(),
                "correlation_id": getattr(request.state, "correlation_id", None),
            },
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        logger.warning(
            "value_error path=%s correlation_id=%s detail=%s",
            request.url.path,
            getattr(request.state, "correlation_id", None),
            str(exc)[:200],
        )
        return JSONResponse(
            status_code=400,
            content={
                "detail": str(exc),
                "correlation_id": getattr(request.state, "correlation_id", None),
            },
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        if exc.status_code >= 500:
            logger.error(
                "http_exception path=%s status=%s correlation_id=%s detail=%s",
                request.url.path,
                exc.status_code,
                getattr(request.state, "correlation_id", None),
                str(exc.detail)[:200],
            )
        elif exc.status_code >= 400:
            logger.warning(
                "http_exception path=%s status=%s correlation_id=%s detail=%s",
                request.url.path,
                exc.status_code,
                getattr(request.state, "correlation_id", None),
                str(exc.detail)[:200],
            )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "detail": exc.detail,
                "correlation_id": getattr(request.state, "correlation_id", None),
            },
            headers=exc.headers,
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception(
            "unhandled_exception path=%s correlation_id=%s",
            request.url.path,
            getattr(request.state, "correlation_id", None),
        )
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error",
                "correlation_id": getattr(request.state, "correlation_id", None),
            },
        )
