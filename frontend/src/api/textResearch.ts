import { apiFetch, type Paginated } from "./client";
import type {
    AnalysisRun,
    Annotation,
    AnnotationLabel,
    AnnotationProgress,
    AnnotationQueueItem,
    Codebook,
    CorpusDocument,
    DashboardSummary,
    DatasetPreview,
    ExportManifest,
    PreprocessingProfile,
    ResearchCorpus,
    TrainedModel,
    TrainingDatasetSnapshot,
    UnitType,
    UncertainPrediction,
} from "../features/text-research/types";

const BASE = "/research";

// ------------------------------------------------------------------
// Corpora
// ------------------------------------------------------------------

export async function seedDemoCorpus(
    projectId: string,
    corpusName?: string
): Promise<ResearchCorpus> {
    return apiFetch(`${BASE}/projects/${projectId}/demo-seed`, {
        method: "POST",
        body: JSON.stringify({ corpus_name: corpusName ?? null }),
    });
}

export async function createCorpus(
    projectId: string,
    payload: { name: string; description?: string }
): Promise<ResearchCorpus> {
    return apiFetch(`${BASE}/projects/${projectId}/corpora`, {
        method: "POST",
        body: JSON.stringify(payload),
    });
}

export async function listCorpora(projectId: string): Promise<ResearchCorpus[]> {
    return apiFetch(`${BASE}/projects/${projectId}/corpora`);
}

export async function getCorpus(corpusId: string): Promise<ResearchCorpus> {
    return apiFetch(`${BASE}/corpora/${corpusId}`);
}

export async function updateCorpus(
    corpusId: string,
    payload: { name?: string; description?: string }
): Promise<ResearchCorpus> {
    return apiFetch(`${BASE}/corpora/${corpusId}`, {
        method: "PATCH",
        body: JSON.stringify(payload),
    });
}

export async function deleteCorpus(corpusId: string): Promise<void> {
    return apiFetch(`${BASE}/corpora/${corpusId}`, { method: "DELETE" });
}

// ------------------------------------------------------------------
// Documents
// ------------------------------------------------------------------

export async function listDocuments(
    corpusId: string,
    params?: { limit?: number; offset?: number }
): Promise<Paginated<CorpusDocument>> {
    const search = new URLSearchParams();
    if (params?.limit != null) search.set("limit", String(params.limit));
    if (params?.offset != null) search.set("offset", String(params.offset));
    const qs = search.toString();
    return apiFetch(`${BASE}/corpora/${corpusId}/documents${qs ? `?${qs}` : ""}`);
}

export async function addDocument(
    corpusId: string,
    payload: {
        rag_document_id: string;
        title?: string;
        organization?: string;
        publication_year?: number;
        country?: string;
        language?: string;
    }
): Promise<CorpusDocument> {
    return apiFetch(`${BASE}/corpora/${corpusId}/documents`, {
        method: "POST",
        body: JSON.stringify(payload),
    });
}

export async function getSourceText(documentId: string): Promise<{ document_id: string; text: string }> {
    return apiFetch(`${BASE}/documents/${documentId}/source-text`);
}

// ------------------------------------------------------------------
// Segmentation
// ------------------------------------------------------------------

export async function segmentCorpus(
    corpusId: string,
    unitType: UnitType
): Promise<AnalysisRun> {
    return apiFetch(`${BASE}/corpora/${corpusId}/segment`, {
        method: "POST",
        body: JSON.stringify({ unit_type: unitType }),
    });
}

// ------------------------------------------------------------------
// Codebooks & labels
// ------------------------------------------------------------------

export async function createCodebook(
    projectId: string,
    payload: { name: string; description?: string; seed_demo_labels?: boolean }
): Promise<Codebook> {
    return apiFetch(`${BASE}/projects/${projectId}/codebooks`, {
        method: "POST",
        body: JSON.stringify({
            name: payload.name,
            description: payload.description,
            seed_demo_labels: payload.seed_demo_labels ?? true,
        }),
    });
}

export async function listCodebooks(projectId: string): Promise<Codebook[]> {
    return apiFetch(`${BASE}/projects/${projectId}/codebooks`);
}

export async function listLabels(codebookId: string): Promise<AnnotationLabel[]> {
    return apiFetch(`${BASE}/codebooks/${codebookId}/labels`);
}

export async function addLabel(
    codebookId: string,
    payload: { name: string; description?: string }
): Promise<AnnotationLabel> {
    return apiFetch(`${BASE}/codebooks/${codebookId}/labels`, {
        method: "POST",
        body: JSON.stringify(payload),
    });
}

export async function updateLabel(
    labelId: string,
    payload: Partial<{
        name: string;
        description: string;
        inclusion_criteria: string;
        exclusion_criteria: string;
    }>
): Promise<AnnotationLabel> {
    return apiFetch(`${BASE}/labels/${labelId}`, {
        method: "PATCH",
        body: JSON.stringify(payload),
    });
}

// ------------------------------------------------------------------
// Annotation
// ------------------------------------------------------------------

export async function listAnnotationQueue(status?: string): Promise<AnnotationQueueItem[]> {
    const qs = status ? `?status=${encodeURIComponent(status)}` : "";
    return apiFetch(`${BASE}/annotations/queue${qs}`);
}

export async function saveAnnotations(payload: {
    text_unit_id: string;
    codebook_id: string;
    values: Array<{ label_id: string; value: string; confidence?: number; comment?: string }>;
    mark_task_complete?: boolean;
}): Promise<Annotation[]> {
    return apiFetch(`${BASE}/annotations`, {
        method: "POST",
        body: JSON.stringify(payload),
    });
}

export async function getAnnotationProgress(corpusId: string): Promise<AnnotationProgress> {
    return apiFetch(`${BASE}/corpora/${corpusId}/annotations/progress`);
}

export async function listUnitAnnotations(textUnitId: string): Promise<Annotation[]> {
    return apiFetch(`${BASE}/text-units/${textUnitId}/annotations`);
}

// ------------------------------------------------------------------
// Reliability
// ------------------------------------------------------------------

export async function computeReliability(
    corpusId: string,
    payload: { codebook_id: string; label_ids?: string[] }
): Promise<AnalysisRun> {
    return apiFetch(`${BASE}/corpora/${corpusId}/reliability`, {
        method: "POST",
        body: JSON.stringify(payload),
    });
}

// ------------------------------------------------------------------
// Analysis
// ------------------------------------------------------------------

export async function runCorpusStats(
    corpusId: string,
    payload: { unit_type: UnitType; preprocessing_profile_id?: string }
): Promise<AnalysisRun> {
    return apiFetch(`${BASE}/corpora/${corpusId}/analysis/corpus-stats`, {
        method: "POST",
        body: JSON.stringify(payload),
    });
}

export async function runFrequencies(
    corpusId: string,
    payload: { unit_type: UnitType; top_n?: number; preprocessing_profile_id?: string }
): Promise<AnalysisRun> {
    return apiFetch(`${BASE}/corpora/${corpusId}/analysis/frequencies`, {
        method: "POST",
        body: JSON.stringify({ top_n: 50, ...payload }),
    });
}

// ------------------------------------------------------------------
// Classification
// ------------------------------------------------------------------

export async function previewDataset(payload: {
    corpus_id: string;
    unit_type: UnitType;
    codebook_id: string;
    label_ids: string[];
    annotation_source?: string;
}): Promise<DatasetPreview> {
    return apiFetch(`${BASE}/classifiers/dataset-preview`, {
        method: "POST",
        body: JSON.stringify({
            annotation_source: "adjudicated_only",
            ...payload,
        }),
    });
}

export async function trainClassifier(payload: {
    snapshot_id: string;
    algorithm?: string;
    name?: string;
    run_async?: boolean;
}): Promise<AnalysisRun> {
    return apiFetch(`${BASE}/classifiers/train`, {
        method: "POST",
        body: JSON.stringify({
            algorithm: "logistic_regression",
            run_async: false,
            ...payload,
        }),
    });
}

export async function predictClassifier(
    modelId: string,
    payload: { unit_type: UnitType; only_unannotated?: boolean }
): Promise<AnalysisRun> {
    return apiFetch(`${BASE}/classifiers/${modelId}/predict`, {
        method: "POST",
        body: JSON.stringify(payload),
    });
}

export async function listUncertainPredictions(
    modelId: string,
    limit = 20
): Promise<UncertainPrediction[]> {
    return apiFetch(`${BASE}/classifiers/${modelId}/active-learning/queue?limit=${limit}`);
}

export async function assignUncertainPredictions(
    modelId: string,
    textUnitIds: string[],
    annotatorIds: string[]
): Promise<unknown> {
    return apiFetch(`${BASE}/classifiers/${modelId}/active-learning/assign`, {
        method: "POST",
        body: JSON.stringify({ text_unit_ids: textUnitIds, annotator_ids: annotatorIds }),
    });
}

export async function listClassifiers(
    projectId: string,
    corpusId?: string
): Promise<TrainedModel[]> {
    const qs = corpusId ? `?corpus_id=${encodeURIComponent(corpusId)}` : "";
    return apiFetch(`${BASE}/projects/${projectId}/classifiers${qs}`);
}

export async function listDatasetSnapshots(
    projectId: string,
    corpusId?: string
): Promise<TrainingDatasetSnapshot[]> {
    const qs = corpusId ? `?corpus_id=${encodeURIComponent(corpusId)}` : "";
    return apiFetch(`${BASE}/projects/${projectId}/dataset-snapshots${qs}`);
}

export async function freezeDataset(payload: {
    name: string;
    corpus_id: string;
    unit_type: UnitType;
    codebook_id: string;
    label_ids: string[];
}): Promise<TrainingDatasetSnapshot> {
    return apiFetch(`${BASE}/classifiers/dataset-snapshots`, {
        method: "POST",
        body: JSON.stringify({
            annotation_source: "adjudicated_only",
            ...payload,
        }),
    });
}

// ------------------------------------------------------------------
// Topics
// ------------------------------------------------------------------

export async function trainTopicModel(
    corpusId: string,
    payload: {
        unit_type: UnitType;
        algorithm?: string;
        n_topics?: number;
        run_async?: boolean;
    }
): Promise<AnalysisRun> {
    return apiFetch(`${BASE}/corpora/${corpusId}/topics/train`, {
        method: "POST",
        body: JSON.stringify({
            algorithm: "lda",
            n_topics: 5,
            run_async: false,
            ...payload,
        }),
    });
}

export async function runRobustnessSweep(payload: {
    snapshot_id: string;
    algorithm?: string;
    seeds?: number[];
    cv_folds?: number;
    run_async?: boolean;
}): Promise<AnalysisRun> {
    return apiFetch(`${BASE}/robustness/sweep`, {
        method: "POST",
        body: JSON.stringify({ algorithm: "logistic_regression", run_async: false, ...payload }),
    });
}

// ------------------------------------------------------------------
// Comparative / Explorer
// ------------------------------------------------------------------

export async function runComparativePrevalence(
    corpusId: string,
    payload: {
        unit_type: UnitType;
        codebook_id: string;
        label_ids: string[];
        group_by: string;
        provenance_mode?: string;
    }
): Promise<AnalysisRun> {
    return apiFetch(`${BASE}/corpora/${corpusId}/comparative/prevalence`, {
        method: "POST",
        body: JSON.stringify({
            provenance_mode: "human_only",
            ...payload,
        }),
    });
}

// ------------------------------------------------------------------
// Dashboard & runs
// ------------------------------------------------------------------

export async function getDashboardSummary(corpusId: string): Promise<DashboardSummary> {
    return apiFetch(`${BASE}/corpora/${corpusId}/dashboard`);
}

export async function listRuns(
    projectId: string,
    params?: { corpus_id?: string; run_type?: string; limit?: number; offset?: number }
): Promise<Paginated<AnalysisRun>> {
    const search = new URLSearchParams();
    if (params?.corpus_id) search.set("corpus_id", params.corpus_id);
    if (params?.run_type) search.set("run_type", params.run_type);
    if (params?.limit != null) search.set("limit", String(params.limit));
    if (params?.offset != null) search.set("offset", String(params.offset));
    const qs = search.toString();
    return apiFetch(`${BASE}/projects/${projectId}/runs${qs ? `?${qs}` : ""}`);
}

export async function getRun(runId: string): Promise<AnalysisRun> {
    return apiFetch(`${BASE}/runs/${runId}`);
}

// ------------------------------------------------------------------
// Exports
// ------------------------------------------------------------------

export async function getExportManifest(corpusId: string): Promise<ExportManifest> {
    return apiFetch(`${BASE}/corpora/${corpusId}/export/manifest`);
}

export async function getQuantedaScript(corpusId: string): Promise<{ script: string }> {
    return apiFetch(`${BASE}/corpora/${corpusId}/export/quanteda-script`);
}

export async function listPreprocessingProfiles(projectId: string): Promise<PreprocessingProfile[]> {
    return apiFetch(`${BASE}/projects/${projectId}/preprocessing-profiles`);
}

export function researchExportUrl(path: string): string {
    const apiBase = import.meta.env.VITE_API_BASE ?? "/api/v1";
    return `${apiBase}${path}`;
}
