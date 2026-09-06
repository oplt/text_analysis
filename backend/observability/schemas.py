from typing import Literal

from pydantic import BaseModel

HealthStatus = Literal["healthy", "degraded", "down", "unknown", "not_configured"]


class ObservabilityToolLink(BaseModel):
    url: str | None
    configured: bool
    allowed: bool


class ObservabilityDashboards(BaseModel):
    application_overview: ObservabilityToolLink
    api: ObservabilityToolLink
    frontend: ObservabilityToolLink
    database: ObservabilityToolLink
    cache: ObservabilityToolLink
    workers: ObservabilityToolLink
    scheduled_tasks: ObservabilityToolLink
    errors: ObservabilityToolLink


class ObservabilityLinks(BaseModel):
    grafana_base_url: ObservabilityToolLink
    prometheus_url: ObservabilityToolLink
    tempo_explore_url: ObservabilityToolLink
    dashboards: ObservabilityDashboards


class ObservabilityStatusItem(BaseModel):
    status: HealthStatus
    detail: str
    url: str | None = None
    value: str | None = None
    queue_depth: int | None = None
    last_checked_at: str | None = None


class ObservabilityStatus(BaseModel):
    api: ObservabilityStatusItem
    frontend: ObservabilityStatusItem
    database: ObservabilityStatusItem
    cache: ObservabilityStatusItem
    workers: ObservabilityStatusItem
    background_jobs: ObservabilityStatusItem
    error_rate: ObservabilityStatusItem
    request_latency: ObservabilityStatusItem
    prometheus: ObservabilityStatusItem
    grafana: ObservabilityStatusItem
    tempo: ObservabilityStatusItem
