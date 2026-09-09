import type { TrainedModel } from "./types";

export function asRecord(value: unknown): Record<string, unknown> | null {
    return value && typeof value === "object" && !Array.isArray(value)
        ? (value as Record<string, unknown>)
        : null;
}

export function num(value: unknown): number | null {
    if (typeof value === "number" && !Number.isNaN(value)) return value;
    if (typeof value === "string" && value.trim() !== "") {
        const parsed = Number(value);
        return Number.isNaN(parsed) ? null : parsed;
    }
    return null;
}

export function formatMetric(value: number | null | undefined, digits = 3): string {
    if (value == null || Number.isNaN(value)) return "—";
    return Number.isInteger(value) ? String(value) : value.toFixed(digits);
}

/** Map backend lifecycle to registry display labels. */
export function lifecycleDisplayLabel(status: string | null | undefined): string {
    switch (status) {
        case "approved":
            return "Production";
        case "deprecated":
            return "Deprecated";
        case "candidate":
            return "Candidate";
        default:
            return status?.trim() || "Unknown";
    }
}

export function lifecycleChipColor(
    status: string | null | undefined
): "success" | "default" | "warning" | "error" {
    switch (status) {
        case "approved":
            return "success";
        case "deprecated":
            return "default";
        case "candidate":
            return "warning";
        default:
            return "default";
    }
}

export function modelDisplayName(model: Pick<TrainedModel, "id" | "name" | "version">): string {
    if (model.name?.trim()) return model.name.trim();
    return `${model.id.slice(0, 8)}-v${model.version}`;
}

export function extractMacroF1(metrics: Record<string, unknown> | null | undefined): number | null {
    if (!metrics) return null;
    return (
        num(metrics.f1_macro) ??
        num(metrics.macro_f1) ??
        num(asRecord(metrics.holdout)?.f1_macro) ??
        num(asRecord(metrics.test)?.f1_macro) ??
        num(asRecord(metrics.cv)?.mean_f1_macro)
    );
}

export function extractConfidenceIntervals(
    metrics: Record<string, unknown> | null | undefined
): Record<string, unknown> | null {
    if (!metrics) return null;
    return (
        asRecord(metrics.confidence_intervals) ??
        asRecord(metrics.bootstrap_cis) ??
        asRecord(metrics.cis) ??
        asRecord(asRecord(metrics.holdout)?.confidence_intervals)
    );
}

function parseJsonish<T>(value: unknown, fallback: T): T {
    if (value == null) return fallback;
    if (typeof value === "string") {
        try {
            return JSON.parse(value) as T;
        } catch {
            return fallback;
        }
    }
    return value as T;
}

export type NormalizedPrediction = {
    predicted_labels: string[];
    scores: Record<string, number>;
    uncertainty: number | null;
};

/** Normalize list-predictions / prediction-set rows (ORM or API schema). */
export function normalizePredictionRow(row: unknown): NormalizedPrediction | null {
    const record = asRecord(row);
    if (!record) return null;
    const labelsRaw =
        record.predicted_labels ?? parseJsonish(record.predicted_labels_json, []);
    const scoresRaw = record.scores ?? parseJsonish(record.scores_json, {});
    const labels = Array.isArray(labelsRaw) ? labelsRaw.map(String) : [];
    const scoresRecord = asRecord(scoresRaw) ?? {};
    const scores: Record<string, number> = {};
    for (const [key, value] of Object.entries(scoresRecord)) {
        const parsed = num(value);
        if (parsed != null) scores[key] = parsed;
    }
    return {
        predicted_labels: labels,
        scores,
        uncertainty: num(record.uncertainty),
    };
}

export function aggregatePredictionsForDrift(rows: unknown[]): {
    label_counts: Record<string, number>;
    scores: number[];
} {
    const label_counts: Record<string, number> = {};
    const scores: number[] = [];
    for (const row of rows) {
        const pred = normalizePredictionRow(row);
        if (!pred) continue;
        if (!pred.predicted_labels.length) {
            label_counts.__unlabeled__ = (label_counts.__unlabeled__ ?? 0) + 1;
        } else {
            for (const label of pred.predicted_labels) {
                label_counts[label] = (label_counts[label] ?? 0) + 1;
            }
        }
        for (const value of Object.values(pred.scores)) {
            scores.push(value);
        }
        if (pred.uncertainty != null) scores.push(pred.uncertainty);
    }
    return { label_counts, scores };
}
