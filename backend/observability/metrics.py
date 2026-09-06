from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Any

from fastapi import Request, Response

from backend.observability.config import load_config

logger = logging.getLogger(__name__)


def _safe_attrs(attrs: dict[str, Any] | None = None) -> dict[str, str | int | float | bool]:
    safe: dict[str, str | int | float | bool] = {}
    for key, value in (attrs or {}).items():
        if value is None:
            continue
        if isinstance(value, str | int | float | bool):
            safe[key] = value
        else:
            safe[key] = str(value)
    return safe


@lru_cache(maxsize=1)
def _meter():
    from opentelemetry import metrics

    return metrics.get_meter("generic_app.telemetry")


@lru_cache(maxsize=1)
def _instruments() -> dict[str, Any]:
    meter = _meter()
    return {
        "api_request_failures": meter.create_counter("generic_app.api.request_failures"),
        "background_jobs_total": meter.create_counter("generic_app.background_jobs_total"),
        "background_job_duration_ms": meter.create_histogram(
            "generic_app.background_job_duration_ms",
            unit="ms",
        ),
    }


def add(name: str, value: int | float = 1, attrs: dict[str, Any] | None = None) -> None:
    try:
        _instruments()[name].add(value, _safe_attrs(attrs))
    except Exception:
        return


def record(name: str, value: int | float, attrs: dict[str, Any] | None = None) -> None:
    try:
        _instruments()[name].record(value, _safe_attrs(attrs))
    except Exception:
        return


def setup_metrics(app: Any) -> None:
    """Expose Prometheus metrics without making app startup depend on Prometheus."""

    config = load_config()
    if not config.prometheus_metrics_enabled:
        logger.info("Prometheus metrics disabled by PROMETHEUS_METRICS_ENABLED")
        return
    if getattr(app.state, "prometheus_metrics_instrumented", False):
        return
    try:
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

        from backend.observability import prometheus_metrics
    except ImportError:
        logger.warning("prometheus-client not installed - Prometheus metrics disabled")
        return

    @app.middleware("http")
    async def prometheus_metrics_middleware(request: Request, call_next: Any) -> Response:
        route = _route_path(request)
        if route == config.prometheus_metrics_path:
            return await call_next(request)

        started_at = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as exc:
            elapsed = time.perf_counter() - started_at
            prometheus_metrics.http_request_duration_seconds.labels(
                request.method,
                route,
            ).observe(elapsed)
            prometheus_metrics.http_requests_total.labels(
                request.method,
                route,
                "500",
            ).inc()
            prometheus_metrics.http_exceptions_total.labels(
                request.method,
                route,
                type(exc).__name__,
            ).inc()
            raise

        elapsed = time.perf_counter() - started_at
        prometheus_metrics.http_request_duration_seconds.labels(
            request.method,
            route,
        ).observe(elapsed)
        prometheus_metrics.http_requests_total.labels(
            request.method,
            route,
            str(response.status_code),
        ).inc()
        return response

    def metrics_endpoint() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    app.add_api_route(
        config.prometheus_metrics_path,
        metrics_endpoint,
        methods=["GET"],
        include_in_schema=False,
    )
    app.state.prometheus_metrics_instrumented = True


def _route_path(request: Request) -> str:
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    if isinstance(path, str) and path:
        return path
    return request.url.path
