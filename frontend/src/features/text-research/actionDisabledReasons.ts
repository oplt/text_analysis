/**
 * Human-readable disabled reasons for primary research actions (Phase 20).
 */

export function trainClassifierDisabledReason(options: {
    snapshotId?: string | null;
    pending?: boolean;
}): string | null {
    if (options.pending) return "Training is already in progress.";
    if (!options.snapshotId) {
        return "Freeze or select a training dataset snapshot before training.";
    }
    return null;
}

export function computeReliabilityDisabledReason(options: {
    corpusId?: string | null;
    codebookId?: string | null;
    labelCount?: number;
    pending?: boolean;
}): string | null {
    if (options.pending) return "Reliability computation is already in progress.";
    if (!options.corpusId) return "Select a corpus before computing reliability.";
    if (!options.codebookId) return "Select a codebook with labels before computing reliability.";
    if ((options.labelCount ?? 0) < 1) {
        return "Add at least one codebook label before computing reliability.";
    }
    return null;
}

/** Shown beside empty/ready copy — reliability also needs overlapping multi-coder work. */
export const RELIABILITY_ANNOTATOR_HINT =
    "Agreement metrics require overlapping annotations from at least two completed annotators on the same units.";

export function segmentCorpusDisabledReason(options: {
    documentCount?: number;
    active?: boolean;
}): string | null {
    if (options.active) return "Segmentation is already running.";
    if ((options.documentCount ?? 0) < 1) {
        return "Upload or link documents in Corpus before starting segmentation.";
    }
    return null;
}

export function analysisRunDisabledReason(options: {
    corpusId?: string | null;
    pending?: boolean;
    tab?: string;
    kwicKeyword?: string;
    kwicSearchMode?: string;
    kwicQueryMode?: string;
    kwicQueryLanguage?: string;
    keynessA?: string;
    keynessB?: string;
    hasDictionaryInput?: boolean;
}): string | null {
    if (!options.corpusId) return "Select a corpus before running analysis.";
    if (options.pending) return "An analysis run is already in progress.";
    if (options.tab === "kwic") {
        if (!options.kwicKeyword?.trim()) return "Enter a KWIC keyword or query before running.";
        if (
            options.kwicSearchMode === "lexical" &&
            options.kwicQueryMode === "lemma" &&
            !options.kwicQueryLanguage?.trim()
        ) {
            return "Lemma KWIC search requires a query language.";
        }
    }
    if (options.tab === "keyness") {
        if (!options.keynessA?.trim() || !options.keynessB?.trim()) {
            return "Set both keyness groups (A and B) before running.";
        }
    }
    if (options.tab === "dictionaries" && !options.hasDictionaryInput) {
        return "Select a dictionary or enter terms before running dictionary analysis.";
    }
    return null;
}

export function robustnessSweepDisabledReason(options: {
    snapshotId?: string | null;
    pending?: boolean;
}): string | null {
    if (options.pending) return "A robustness sweep is already running.";
    if (!options.snapshotId) {
        return "Select a frozen training dataset snapshot before running the sweep.";
    }
    return null;
}

export function comparativeRunDisabledReason(options: {
    corpusId?: string | null;
    codebookId?: string | null;
    pending?: boolean;
}): string | null {
    if (options.pending) return "Comparative analysis is already running.";
    if (!options.corpusId) return "Select a corpus before running comparative prevalence.";
    if (!options.codebookId) return "Select a codebook before running comparative prevalence.";
    return null;
}

export function agentRunDisabledReason(options: {
    message?: string;
    pending?: boolean;
}): string | null {
    if (options.pending) return "An agent run is already in progress.";
    if (!options.message?.trim()) return "Enter a task message before running the agent.";
    return null;
}

export function predictModelDisabledReason(options: {
    modelId?: string | null;
    pending?: boolean;
}): string | null {
    if (options.pending) return "Prediction is already in progress.";
    if (!options.modelId) return "Select a trained model before predicting.";
    return null;
}

export function trainTopicModelDisabledReason(options: {
    corpusId?: string | null;
    pending?: boolean;
}): string | null {
    if (options.pending) return "Topic model training is already in progress.";
    if (!options.corpusId) return "Select a corpus before training a topic model.";
    return null;
}

export function assignUncertainDisabledReason(options: {
    modelId?: string | null;
    selectedUnitCount?: number;
    annotatorCount?: number;
    pending?: boolean;
}): string | null {
    if (options.pending) return "Assignment is already in progress.";
    if (!options.modelId) return "Select a trained model before assigning units.";
    if ((options.selectedUnitCount ?? 0) < 1) {
        return "Select one or more uncertain units to assign.";
    }
    if ((options.annotatorCount ?? 0) < 1) {
        return "Select at least one annotator before assigning.";
    }
    return null;
}

export function modelLifecycleDisabledReason(options: {
    action: "production" | "deprecated" | "archived";
    currentStatus?: string | null;
    pending?: boolean;
}): string | null {
    if (options.pending) return "A lifecycle update is already in progress.";
    const status = options.currentStatus ?? "";
    if (options.action === "production" && status === "production") {
        return "This model is already in production.";
    }
    if (options.action === "deprecated" && status === "deprecated") {
        return "This model is already deprecated.";
    }
    if (options.action === "archived" && status === "archived") {
        return "This model is already archived.";
    }
    return null;
}

export function driftCompareDisabledReason(options: {
    baselineId?: string | null;
    currentId?: string | null;
    pending?: boolean;
}): string | null {
    if (options.pending) return "Drift comparison is already running.";
    if (!options.baselineId || !options.currentId) {
        return "Select baseline and current prediction sets before comparing.";
    }
    return null;
}
