import type { GrafanaUrlContext, ObservabilityStatus, ObservabilityStatusItem } from "./types";

export type HealthKey =
    | "api" | "frontend" | "database" | "cache" | "workers" | "backgroundJobs"
    | "errorRate" | "requestLatency" | "prometheus" | "grafana" | "tempo";

export const SERVICES = ["backend", "frontend", "worker"];
export const ENVIRONMENTS = ["local", "development", "staging", "production"];
export const TIME_RANGES = [
    { label: "Last 15 minutes", from: "now-15m", to: "now" },
    { label: "Last hour", from: "now-1h", to: "now" },
    { label: "Last 6 hours", from: "now-6h", to: "now" },
    { label: "Last 24 hours", from: "now-24h", to: "now" },
];

export function buildHealthItems(status?: ObservabilityStatus): Record<HealthKey, ObservabilityStatusItem | undefined> {
    return {
        api: status?.api,
        frontend: status?.frontend,
        database: status?.database,
        cache: status?.cache,
        workers: status?.workers,
        backgroundJobs: status?.background_jobs,
        errorRate: status?.error_rate,
        requestLatency: status?.request_latency,
        prometheus: status?.prometheus,
        grafana: status?.grafana,
        tempo: status?.tempo,
    };
}

export function metricFor(item?: ObservabilityStatusItem) {
    return item?.value ??
        (item?.queue_depth !== undefined && item.queue_depth !== null ? `${item.queue_depth} queued` : null);
}

export function fallbackItem(detail: string): ObservabilityStatusItem {
    return { status: "unknown", detail };
}

export type ObservabilityFilters = {
    service: string;
    environment: string;
    route: string;
    jobName: string;
    traceId: string;
    requestId: string;
    timeRangeIndex: number;
};

export function buildObservabilityContext(values: ObservabilityFilters): GrafanaUrlContext {
    const timeRange = TIME_RANGES[values.timeRangeIndex] ?? TIME_RANGES[1];
    return {
        service: values.service,
        environment: values.environment,
        route: values.route,
        jobName: values.jobName,
        traceId: values.traceId,
        requestId: values.requestId,
        from: timeRange.from,
        to: timeRange.to,
        orgId: "1",
    };
}
