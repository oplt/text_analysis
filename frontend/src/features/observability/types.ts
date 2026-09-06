export type HealthStatus = "healthy" | "degraded" | "down" | "unknown" | "not_configured";

export type ObservabilityToolLink = {
    url: string | null;
    configured: boolean;
    allowed: boolean;
};

export type ObservabilityDashboards = {
    application_overview: ObservabilityToolLink;
    api: ObservabilityToolLink;
    frontend: ObservabilityToolLink;
    database: ObservabilityToolLink;
    cache: ObservabilityToolLink;
    workers: ObservabilityToolLink;
    scheduled_tasks: ObservabilityToolLink;
    errors: ObservabilityToolLink;
};

export type ObservabilityLinks = {
    grafana_base_url: ObservabilityToolLink;
    prometheus_url: ObservabilityToolLink;
    tempo_explore_url: ObservabilityToolLink;
    dashboards: ObservabilityDashboards;
};

export type ObservabilityStatusItem = {
    status: HealthStatus;
    detail: string;
    url?: string | null;
    value?: string | null;
    queue_depth?: number | null;
    last_checked_at?: string | null;
};

export type ObservabilityStatus = {
    api: ObservabilityStatusItem;
    frontend: ObservabilityStatusItem;
    database: ObservabilityStatusItem;
    cache: ObservabilityStatusItem;
    workers: ObservabilityStatusItem;
    background_jobs: ObservabilityStatusItem;
    error_rate: ObservabilityStatusItem;
    request_latency: ObservabilityStatusItem;
    prometheus: ObservabilityStatusItem;
    grafana: ObservabilityStatusItem;
    tempo: ObservabilityStatusItem;
};

export type GrafanaUrlContext = {
    service?: string;
    environment?: string;
    route?: string;
    jobName?: string;
    traceId?: string;
    requestId?: string;
    from?: string;
    to?: string;
    orgId?: string;
};
