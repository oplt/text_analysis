"""Prometheus metric definitions for the FastAPI backend."""

from prometheus_client import Counter, Gauge, Histogram

http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "route", "status_code"],
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "route"],
)

http_exceptions_total = Counter(
    "http_exceptions_total",
    "Total unhandled HTTP exceptions",
    ["method", "route", "exception_type"],
)

database_pool_checked_out = Gauge(
    "database_pool_checked_out_connections",
    "SQLAlchemy connections currently checked out from the process pool",
    ["role"],
)
database_pool_wait_seconds = Histogram(
    "database_pool_wait_seconds",
    "Observed SQLAlchemy connection checkout wait time",
    ["role"],
)
database_pool_timeouts_total = Counter(
    "database_pool_timeouts_total",
    "SQLAlchemy pool checkout timeouts",
    ["role"],
)
database_session_duration_seconds = Histogram(
    "database_session_duration_seconds",
    "Async SQLAlchemy session lifetime",
    ["role"],
)
