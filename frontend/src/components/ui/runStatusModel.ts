/**
 * Canonical run / job status model (Phase 16).
 * One label + color system for research runs, RAG ingestion, and agent workflows.
 */

export type CanonicalRunStatus =
    | "queued"
    | "running"
    | "completed"
    | "failed"
    | "cancelled"
    | "unknown";

export type RunStatusTone = "default" | "info" | "success" | "warning" | "error";

const QUEUED = new Set(["queued", "pending", "created", "uploaded"]);
const RUNNING = new Set([
    "running",
    "processing",
    "in_progress",
    "in-progress",
    "started",
    "active",
]);
const COMPLETED = new Set(["completed", "complete", "success", "succeeded", "indexed", "done"]);
const FAILED = new Set(["failed", "error", "errored"]);
const CANCELLED = new Set(["cancelled", "canceled"]);

export function normalizeRunStatus(raw: string | null | undefined): CanonicalRunStatus {
    const status = (raw ?? "").trim().toLowerCase();
    if (!status) return "unknown";
    if (QUEUED.has(status)) return "queued";
    if (RUNNING.has(status)) return "running";
    if (COMPLETED.has(status)) return "completed";
    if (FAILED.has(status)) return "failed";
    if (CANCELLED.has(status)) return "cancelled";
    return "unknown";
}

export function runStatusLabel(raw: string | null | undefined): string {
    switch (normalizeRunStatus(raw)) {
        case "queued":
            return "Queued";
        case "running":
            return "Running";
        case "completed":
            return "Completed";
        case "failed":
            return "Failed";
        case "cancelled":
            return "Cancelled";
        default:
            return raw?.trim() ? raw : "Unknown";
    }
}

export function runStatusTone(raw: string | null | undefined): RunStatusTone {
    switch (normalizeRunStatus(raw)) {
        case "queued":
            return "info";
        case "running":
            return "warning";
        case "completed":
            return "success";
        case "failed":
            return "error";
        case "cancelled":
            return "default";
        default:
            return "default";
    }
}

export function isActiveCanonicalStatus(raw: string | null | undefined): boolean {
    const status = normalizeRunStatus(raw);
    return status === "queued" || status === "running";
}

export function formatRunDurationMs(ms: number | null | undefined): string {
    if (ms == null || !Number.isFinite(ms) || ms < 0) return "—";
    if (ms < 1000) return `${Math.round(ms)} ms`;
    const seconds = ms / 1000;
    if (seconds < 60) return `${seconds.toFixed(1)} s`;
    const minutes = Math.floor(seconds / 60);
    const rem = Math.round(seconds % 60);
    return `${minutes}m ${rem}s`;
}

export function runDurationMs(options: {
    startedAt?: string | null;
    completedAt?: string | null;
    createdAt?: string | null;
    latencyMs?: number | null;
    /** When still active, compute elapsed from start → now. */
    nowMs?: number;
}): number | null {
    if (options.latencyMs != null && Number.isFinite(options.latencyMs)) {
        return options.latencyMs;
    }
    const startRaw = options.startedAt ?? options.createdAt;
    if (!startRaw) return null;
    const start = Date.parse(startRaw);
    if (!Number.isFinite(start)) return null;
    const endRaw = options.completedAt;
    const end = endRaw ? Date.parse(endRaw) : options.nowMs ?? Date.now();
    if (!Number.isFinite(end) || end < start) return null;
    return end - start;
}

export function formatRunTimestamp(iso: string | null | undefined): string {
    if (!iso) return "—";
    const ms = Date.parse(iso);
    if (!Number.isFinite(ms)) return "—";
    return new Date(ms).toLocaleString();
}
