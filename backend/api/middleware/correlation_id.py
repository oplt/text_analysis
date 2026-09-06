import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from backend.core.log_context import reset_correlation_id, set_correlation_id

CORRELATION_ID_HEADER = "X-Correlation-ID"
REQUEST_ID_HEADER = "X-Request-ID"
REQUEST_ID_HEADERS = (CORRELATION_ID_HEADER, REQUEST_ID_HEADER)


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        correlation_id = next(
            (request.headers.get(header) for header in REQUEST_ID_HEADERS if request.headers.get(header)),
            None,
        ) or str(uuid.uuid4())
        request.state.correlation_id = correlation_id
        context_token = set_correlation_id(correlation_id)
        try:
            response = await call_next(request)
        finally:
            reset_correlation_id(context_token)
        response.headers[CORRELATION_ID_HEADER] = correlation_id
        response.headers[REQUEST_ID_HEADER] = correlation_id
        return response
