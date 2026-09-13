import type { DashboardSummary } from "./types";

export function totalTextUnits(summary: DashboardSummary): number {
    return Object.values(summary.text_unit_counts).reduce((sum, count) => sum + count, 0);
}

export function latestRunStatusLabel(summary: DashboardSummary): {
    value: string;
    description: string;
    color: "primary" | "secondary" | "success" | "warning" | "error" | "info";
} {
    const byStatus = summary.analysis_run_counts_by_status ?? {};
    const recent = summary.recent_runs?.[0];
    if (recent) {
        const status = recent.status;
        const color =
            status === "completed"
                ? "success"
                : status === "failed"
                  ? "error"
                  : status === "running" || status === "pending"
                    ? "warning"
                    : "info";
        return {
            value: status,
            description: `${formatRunType(recent.run_type)} · ${recent.id.slice(0, 8)}…`,
            color,
        };
    }

    const priority = ["running", "pending", "failed", "completed", "cancelled"] as const;
    for (const status of priority) {
        const count = byStatus[status] ?? 0;
        if (count > 0) {
            const color =
                status === "completed"
                    ? "success"
                    : status === "failed"
                      ? "error"
                      : status === "running" || status === "pending"
                        ? "warning"
                        : "info";
            return {
                value: status,
                description: `${count} run${count === 1 ? "" : "s"} ${status}`,
                color,
            };
        }
    }

    return {
        value: "None",
        description: "No analysis runs yet",
        color: "secondary",
    };
}

export function formatRunType(runType: string): string {
    return runType.replaceAll("_", " ");
}

export function formatPercent(rate: number, digits = 0): string {
    if (!Number.isFinite(rate)) return "—";
    return `${Math.round(rate * 100 * 10 ** digits) / 10 ** digits}%`;
}

export function formatMetricNumber(value: number | null | undefined, digits = 2): string {
    if (value == null || !Number.isFinite(value)) return "—";
    return value.toFixed(digits);
}

export function pickReliabilityHighlights(
    metrics: Record<string, unknown> | null | undefined
): Array<{ label: string; value: string }> {
    if (!metrics) return [];
    const items: Array<{ label: string; value: string }> = [];
    const meanAlpha = metrics.mean_krippendorff_alpha;
    const meanFleiss = metrics.mean_fleiss_kappa;
    const meanCohen = metrics.mean_cohens_kappa;
    if (typeof meanAlpha === "number") {
        items.push({ label: "Mean Krippendorff α", value: formatMetricNumber(meanAlpha) });
    }
    if (typeof meanFleiss === "number") {
        items.push({ label: "Mean Fleiss' κ", value: formatMetricNumber(meanFleiss) });
    }
    if (typeof meanCohen === "number") {
        items.push({ label: "Mean Cohen's κ", value: formatMetricNumber(meanCohen) });
    }
    const labelsEvaluated = metrics.labels_evaluated;
    if (typeof labelsEvaluated === "number") {
        items.push({ label: "Labels evaluated", value: String(labelsEvaluated) });
    }
    return items.slice(0, 4);
}

export function languageSummary(counts: Record<string, number> | undefined): string {
    if (!counts || Object.keys(counts).length === 0) return "No language metadata yet";
    const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]);
    const top = entries.slice(0, 3).map(([lang, count]) => `${lang} (${count})`);
    const extra = entries.length - top.length;
    return extra > 0 ? `${top.join(", ")} +${extra} more` : top.join(", ");
}

export function workspaceHrefForRunType(projectId: string, runType: string): string {
    const normalized = runType.toLowerCase();
    if (normalized.includes("segment") || normalized.includes("preparation")) {
        return `/research/${projectId}/prepare`;
    }
    if (normalized.includes("train") || normalized.includes("classif")) {
        return `/research/${projectId}/classification`;
    }
    if (normalized.includes("export")) {
        return `/research/${projectId}/exports`;
    }
    if (normalized.includes("reliab")) {
        return `/research/${projectId}/reliability`;
    }
    if (normalized.includes("topic")) {
        return `/research/${projectId}/topics`;
    }
    return `/research/${projectId}/runs`;
}
