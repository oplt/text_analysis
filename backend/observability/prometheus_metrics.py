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
database_pool_size = Gauge(
    "database_pool_size",
    "Configured SQLAlchemy pool size (steady-state connections)",
    ["role"],
)
database_pool_overflow = Gauge(
    "database_pool_overflow",
    "Current SQLAlchemy pool overflow connections in use",
    ["role"],
)
database_pool_saturation = Gauge(
    "database_pool_saturation_ratio",
    "Checked-out connections divided by (pool_size + max_overflow)",
    ["role"],
)
database_pool_wait_seconds = Histogram(
    "database_pool_wait_seconds",
    "Observed SQLAlchemy connection checkout wait time",
    ["role"],
    buckets=(0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 15.0),
)
database_pool_timeouts_total = Counter(
    "database_pool_timeouts_total",
    "SQLAlchemy pool checkout timeouts",
    ["role"],
)
database_pool_connection_failures_total = Counter(
    "database_pool_connection_failures_total",
    "SQLAlchemy pool connection acquisition or invalidation failures",
    ["role", "reason"],
)
database_session_duration_seconds = Histogram(
    "database_session_duration_seconds",
    "Async SQLAlchemy session lifetime",
    ["role"],
)
database_query_duration_seconds = Histogram(
    "database_query_duration_seconds",
    "SQLAlchemy cursor execute duration (sync driver)",
    ["role"],
    buckets=(0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

# Research / cache / worker observability (Phase 22)
research_stage_cache_hits_total = Counter(
    "research_stage_cache_hits_total",
    "Research layered stage-cache hits",
    ["stage"],
)
research_stage_cache_misses_total = Counter(
    "research_stage_cache_misses_total",
    "Research layered stage-cache misses",
    ["stage"],
)
research_prepared_corpus_cache_hits_total = Counter(
    "research_prepared_corpus_cache_hits_total",
    "Prepared-corpus stage cache hits",
)
research_prepared_corpus_cache_misses_total = Counter(
    "research_prepared_corpus_cache_misses_total",
    "Prepared-corpus stage cache misses",
)
research_preprocessing_duration_seconds = Histogram(
    "research_preprocessing_duration_seconds",
    "Prepared-corpus / quantitative preprocessing duration",
    ["path"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 15.0, 60.0),
)
research_sse_active_connections = Gauge(
    "research_sse_active_connections",
    "Active research analysis-run SSE streams",
)

worker_task_queue_delay_seconds = Histogram(
    "worker_task_queue_delay_seconds",
    "Delay between Celery enqueue and task start",
    ["task"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 15.0, 60.0, 300.0),
)
worker_task_duration_seconds = Histogram(
    "worker_task_duration_seconds",
    "Celery task execution duration",
    ["task", "state"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 15.0, 60.0, 300.0, 900.0),
)
worker_tasks_total = Counter(
    "worker_tasks_total",
    "Celery tasks completed",
    ["task", "state"],
)
