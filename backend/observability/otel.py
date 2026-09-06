from __future__ import annotations

import logging
from typing import Any

from backend.observability.config import load_config

logger = logging.getLogger(__name__)

_configured = False
_libraries_instrumented = False
_SIGNALS = {"traces", "metrics", "logs"}


def _signal_endpoint(signal: str, base_endpoint: str) -> str:
    endpoint = base_endpoint.rstrip("/")
    parts = endpoint.split("/")
    if len(parts) >= 2 and parts[-2] == "v1" and parts[-1] in _SIGNALS:
        return "/".join([*parts[:-1], signal])
    return f"{endpoint}/v1/{signal}"


def _resource(service_name: str, environment: str):
    from opentelemetry.sdk.resources import Resource

    return Resource.create(
        {
            "service.name": service_name,
            "deployment.environment": environment,
            "service.namespace": "generic-app",
        }
    )


def setup_observability(app: Any | None = None, *, service_name: str | None = None) -> None:
    """Configure Prometheus metrics, OTLP traces, and app instrumentation."""

    from backend.observability.metrics import setup_metrics

    if app is not None:
        setup_metrics(app)
    setup_traces(app=app, service_name=service_name)


def setup_traces(app: Any | None = None, *, service_name: str | None = None) -> None:
    """Configure OTLP traces. Export failures must not affect app startup."""

    global _configured
    if _configured:
        if app is not None:
            _instrument_app(app)
        return

    config = load_config()
    if not config.enabled:
        logger.info("Observability disabled by OBSERVABILITY_ENABLED")
        return
    if config.traces_exporter in {"", "none", "false", "0", "off"} or not config.otlp_endpoint:
        logger.info("OpenTelemetry traces disabled by OTEL_TRACES_EXPORTER")
        return
    if config.traces_exporter != "otlp":
        logger.warning(
            "Unsupported OTEL_TRACES_EXPORTER=%s; tracing disabled",
            config.traces_exporter,
        )
        return

    resolved_service_name = service_name or config.service_name

    try:
        resource = _resource(resolved_service_name, config.environment)
        _setup_trace_exporter(config, resource)
        _setup_library_instrumentation()
        if app is not None:
            _instrument_app(app)
        _configured = True
        logger.info(
            "OpenTelemetry configured endpoint=%s service=%s",
            config.otlp_endpoint,
            resolved_service_name,
        )
    except ImportError as exc:
        logger.warning("OpenTelemetry packages unavailable; telemetry disabled: %s", exc)
    except Exception as exc:
        logger.warning("OpenTelemetry setup failed; app continues without OTLP export: %s", exc)


def _setup_trace_exporter(config: Any, resource: Any) -> None:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=resource)
    if config.otlp_protocol == "grpc":
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        exporter = OTLPSpanExporter(endpoint=config.otlp_endpoint, insecure=config.otlp_insecure)
    else:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        exporter = OTLPSpanExporter(endpoint=_signal_endpoint("traces", config.otlp_endpoint))

    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)


def _setup_library_instrumentation() -> None:
    global _libraries_instrumented
    if _libraries_instrumented:
        return
    try:
        from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

        from backend.db.session import engine

        SQLAlchemyInstrumentor().instrument(engine=engine.sync_engine)
    except Exception as exc:
        logger.debug("SQLAlchemy OpenTelemetry instrumentation skipped: %s", exc)
    _libraries_instrumented = True


def _instrument_app(app: Any) -> None:
    if getattr(app.state, "otel_instrumented", False):
        return
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app)
        app.state.otel_instrumented = True
    except Exception as exc:
        logger.debug("FastAPI OpenTelemetry instrumentation skipped: %s", exc)
