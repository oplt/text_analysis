/** Helpers for segmentation / prepare workspace progress copy. */

import type { AnalysisRun } from "./types";

const ACTIVE_STATUSES = new Set(["queued", "pending", "running"]);

export function isSegmentationRunActive(run: AnalysisRun | null | undefined): boolean {
    return Boolean(run && ACTIVE_STATUSES.has(run.status));
}

export function segmentationStageLabel(stage: string | null | undefined): string {
    switch (stage) {
        case "queued":
            return "Queued";
        case "preparing":
            return "Preparing corpus";
        case "segmenting":
        case "segmenting_documents":
            return "Segmenting documents";
        case "completed":
            return "Completed";
        case "failed":
            return "Failed";
        default:
            return stage ? stage.replaceAll("_", " ") : "Waiting";
    }
}

export function segmentationProgressLines(run: AnalysisRun | null | undefined): string[] {
    if (!run) return [];

    const lines: string[] = [segmentationStageLabel(run.progress_stage)];
    const metrics = (run.metrics ?? {}) as Record<string, unknown>;
    const documentsTotal = Number(metrics.documents_total ?? run.parameters?.document_count ?? 0);
    const documentsDone = Number(metrics.documents_segmented ?? 0);
    const unitsCreated = Number(metrics.text_units_created ?? 0);
    const unitType = String(metrics.unit_type ?? run.parameters?.unit_type ?? "unit");

    if (documentsTotal > 0) {
        lines.push(`${documentsDone.toLocaleString()} / ${documentsTotal.toLocaleString()} documents`);
    }
    if (unitsCreated > 0 || run.status === "completed") {
        lines.push(
            `${unitsCreated.toLocaleString()} ${unitType} unit${unitsCreated === 1 ? "" : "s"} created`
        );
    }
    return lines;
}

export function segmentationRunStorageKey(corpusId: string): string {
    return `text-research:segmentation-run:${corpusId}`;
}
