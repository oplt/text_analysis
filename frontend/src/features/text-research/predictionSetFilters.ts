import type { AdjudicationRecord } from "../../api/textResearch";
import type { Annotation, ModelPredictionItem } from "./types";
import { asRecord, num } from "./modelRegistryUtils";

export const PREDICTION_LAYER_LABELS = {
    model: "MODEL PREDICTION",
    human: "HUMAN ANNOTATION",
    gold: "ADJUDICATED GOLD",
} as const;

export type ReviewStatus = "unreviewed" | "annotated" | "adjudicated";

export type PredictionRowView = {
    id: string;
    text_unit_id: string;
    predicted_labels: string[];
    scores: Record<string, number>;
    max_confidence: number | null;
    uncertainty: number | null;
    created_at: string;
    human_values: string[];
    gold_values: string[];
    review_status: ReviewStatus;
    human_disagreement: boolean;
    model_vs_gold_disagreement: boolean;
};

export type PredictionSetFilters = {
    predictedLabel: string;
    minConfidence: string;
    maxUncertainty: string;
    reviewStatus: "" | ReviewStatus;
    humanDisagreementOnly: boolean;
};

export const DEFAULT_PREDICTION_SET_FILTERS: PredictionSetFilters = {
    predictedLabel: "",
    minConfidence: "",
    maxUncertainty: "",
    reviewStatus: "",
    humanDisagreementOnly: false,
};

export function maxConfidence(scores: Record<string, number>): number | null {
    const values = Object.values(scores).filter((v) => typeof v === "number" && !Number.isNaN(v));
    if (!values.length) return null;
    return Math.max(...values);
}

export function normalizePredictionItem(row: unknown): ModelPredictionItem | null {
    const record = asRecord(row);
    if (!record || typeof record.id !== "string" || typeof record.text_unit_id !== "string") {
        return null;
    }
    const labelsRaw = record.predicted_labels;
    const labels = Array.isArray(labelsRaw) ? labelsRaw.map(String) : [];
    const scoresRecord = asRecord(record.scores) ?? {};
    const scores: Record<string, number> = {};
    for (const [key, value] of Object.entries(scoresRecord)) {
        const parsed = num(value);
        if (parsed != null) scores[key] = parsed;
    }
    return {
        id: record.id,
        trained_model_id: String(record.trained_model_id ?? ""),
        text_unit_id: record.text_unit_id,
        predicted_labels: labels,
        scores,
        uncertainty: num(record.uncertainty),
        created_at: String(record.created_at ?? ""),
    };
}

function uniqueSorted(values: Iterable<string>): string[] {
    return [...new Set([...values].map(String).filter(Boolean))].sort();
}

export function buildPredictionRows(args: {
    predictions: unknown[];
    annotations: Annotation[];
    adjudications: AdjudicationRecord[];
}): PredictionRowView[] {
    const annotationsByUnit = new Map<string, Annotation[]>();
    for (const row of args.annotations) {
        const list = annotationsByUnit.get(row.text_unit_id) ?? [];
        list.push(row);
        annotationsByUnit.set(row.text_unit_id, list);
    }
    const goldByUnit = new Map<string, string[]>();
    for (const row of args.adjudications) {
        const list = goldByUnit.get(row.text_unit_id) ?? [];
        list.push(row.final_value);
        goldByUnit.set(row.text_unit_id, list);
    }

    const rows: PredictionRowView[] = [];
    for (const raw of args.predictions) {
        const pred = normalizePredictionItem(raw);
        if (!pred) continue;
        const humanRows = annotationsByUnit.get(pred.text_unit_id) ?? [];
        const humanValues = uniqueSorted(humanRows.map((a) => a.value));
        const goldValues = uniqueSorted(goldByUnit.get(pred.text_unit_id) ?? []);
        const byLabel = new Map<string, Set<string>>();
        for (const annotation of humanRows) {
            const set = byLabel.get(annotation.label_id) ?? new Set();
            set.add(annotation.value);
            byLabel.set(annotation.label_id, set);
        }
        const human_disagreement = [...byLabel.values()].some((set) => set.size > 1);

        let review_status: ReviewStatus = "unreviewed";
        if (goldValues.length) review_status = "adjudicated";
        else if (humanValues.length) review_status = "annotated";

        const predSet = new Set(pred.predicted_labels);
        const goldSet = new Set(goldValues);
        const model_vs_gold_disagreement =
            goldSet.size > 0 &&
            (predSet.size !== goldSet.size || [...predSet].some((label) => !goldSet.has(label)));

        rows.push({
            id: pred.id,
            text_unit_id: pred.text_unit_id,
            predicted_labels: pred.predicted_labels,
            scores: pred.scores,
            max_confidence: maxConfidence(pred.scores),
            uncertainty: pred.uncertainty,
            created_at: pred.created_at,
            human_values: humanValues,
            gold_values: goldValues,
            review_status,
            human_disagreement,
            model_vs_gold_disagreement,
        });
    }
    return rows;
}

export function filterPredictionRows(
    rows: PredictionRowView[],
    filters: PredictionSetFilters
): PredictionRowView[] {
    const minConf = filters.minConfidence.trim() === "" ? null : Number(filters.minConfidence);
    const maxUnc = filters.maxUncertainty.trim() === "" ? null : Number(filters.maxUncertainty);
    const label = filters.predictedLabel.trim().toLowerCase();

    return rows.filter((row) => {
        if (label && !row.predicted_labels.some((item) => item.toLowerCase().includes(label))) {
            return false;
        }
        if (minConf != null && !Number.isNaN(minConf)) {
            if (row.max_confidence == null || row.max_confidence < minConf) return false;
        }
        if (maxUnc != null && !Number.isNaN(maxUnc)) {
            if (row.uncertainty == null || row.uncertainty > maxUnc) return false;
        }
        if (filters.reviewStatus && row.review_status !== filters.reviewStatus) return false;
        if (filters.humanDisagreementOnly && !row.human_disagreement) return false;
        return true;
    });
}

/** Explicit: predictions never mutate gold/human arrays (reference isolation). */
export function layersRemainSeparate(
    model: string[],
    human: string[],
    gold: string[]
): { model: string[]; human: string[]; gold: string[] } {
    return {
        model: [...model],
        human: [...human],
        gold: [...gold],
    };
}
