"""Observability runtime configuration.

OTLP endpoint resolution:
- Prefer ``OTEL_EXPORTER_OTLP_ENDPOINT`` (OpenTelemetry standard).
- Fall back to legacy ``OTLP_ENDPOINT`` when the OTEL variable is unset.
- ``OTEL_EXPORTER_OTLP_PROTOCOL`` defaults to ``http/protobuf``; legacy-only
  ``OTLP_ENDPOINT`` without OTEL endpoint implies gRPC for backward compatibility.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from backend.core.config import settings


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


@dataclass(frozen=True)
class ObservabilityConfig:
    enabled: bool = True
    service_name: str = "fastapi-backend"
    environment: str = "local"
    otlp_endpoint: str = "http://localhost:4318"
    otlp_protocol: str = "http/protobuf"
    otlp_insecure: bool = True
    traces_exporter: str = "otlp"
    metrics_exporter: str = "none"
    prometheus_metrics_enabled: bool = True
    prometheus_metrics_path: str = "/metrics"


def load_config() -> ObservabilityConfig:
    legacy_endpoint = settings.OTLP_ENDPOINT.strip()
    configured_endpoint = settings.OTEL_EXPORTER_OTLP_ENDPOINT.strip() or legacy_endpoint
    configured_protocol = (
        "grpc"
        if legacy_endpoint and not settings.OTEL_EXPORTER_OTLP_ENDPOINT
        else settings.OTEL_EXPORTER_OTLP_PROTOCOL
    )

    return ObservabilityConfig(
        enabled=env_bool("OBSERVABILITY_ENABLED", True),
        service_name=env_str("OTEL_SERVICE_NAME", settings.OTEL_SERVICE_NAME or settings.APP_NAME),
        environment=settings.APP_ENV or env_str("APP_ENV", "local"),
        otlp_endpoint=env_str("OTEL_EXPORTER_OTLP_ENDPOINT", configured_endpoint or "http://localhost:4318").rstrip("/"),
        otlp_protocol=env_str("OTEL_EXPORTER_OTLP_PROTOCOL", configured_protocol).lower(),
        otlp_insecure=env_bool("OTLP_INSECURE", settings.OTLP_INSECURE),
        traces_exporter=env_str("OTEL_TRACES_EXPORTER", settings.OTEL_TRACES_EXPORTER).lower(),
        metrics_exporter=env_str("OTEL_METRICS_EXPORTER", "none").lower(),
        prometheus_metrics_enabled=env_bool("PROMETHEUS_METRICS_ENABLED", True),
        prometheus_metrics_path=env_str("PROMETHEUS_METRICS_PATH", "/metrics"),
    )
