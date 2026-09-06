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
    params?: {
        limit?: number;
        offset?: number;
        organization?: string;
        publication_year?: number;
        region?: string;
        cultural_sphere?: string;
        language?: string;
        search?: string;
        sort_by?: string;
        sort_dir?: "asc" | "desc";
    }
): Promise<Paginated<CorpusDocument>> {
    const search = new URLSearchParams();
    if (params?.limit != null) search.set("limit", String(params.limit));
    if (params?.offset != null) search.set("offset", String(params.offset));
    if (params?.organization) search.set("organization", params.organization);
    if (params?.publication_year != null) {
        search.set("publication_year", String(params.publication_year));
    }
    if (params?.region) search.set("region", params.region);
    if (params?.cultural_sphere) search.set("cultural_sphere", params.cultural_sphere);
    if (params?.language) search.set("language", params.language);
    if (params?.search) search.set("search", params.search);
    if (params?.sort_by) search.set("sort_by", params.sort_by);
    if (params?.sort_dir) search.set("sort_dir", params.sort_dir);
    const qs = search.toString();
    return apiFetch(`${BASE}/corpora/${corpusId}/documents${qs ? `?${qs}` : ""}`);
}

export async function addDocument(
    corpusId: string,
    payload: {
        rag_document_id: string;
        title?: string;
        organization?: string;
        organization_type?: string;
        publication_year?: number;
        publication_type?: string;
        country?: string;
        region?: string;
        cultural_sphere?: string;
        language?: string;
        education_level?: string;
        source_url?: string;
        research_notes?: string;
    }
): Promise<CorpusDocument> {
    return apiFetch(`${BASE}/corpora/${corpusId}/documents`, {
        method: "POST",
        body: JSON.stringify(payload),
    });
}

export async function updateDocument(
    documentId: string,
    payload: Partial<{
        title: string;
        organization: string;
        organization_type: string;
        publication_year: number | null;
        publication_type: string;
        country: string;
        region: string;
        cultural_sphere: string;
        language: string;
        education_level: string;
        source_url: string;
        research_notes: string;
        metadata_json: Record<string, unknown>;
    }>
): Promise<CorpusDocument> {
    return apiFetch(`${BASE}/documents/${documentId}`, {
        method: "PATCH",
        body: JSON.stringify(payload),
    });
}

export async function deleteDocument(documentId: string): Promise<void> {
    return apiFetch(`${BASE}/documents/${documentId}`, { method: "DELETE" });
}

export async function bulkUpdateDocumentMetadata(
    corpusId: string,
    payload: { document_ids: string[]; fields: Record<string, unknown> }
): Promise<CorpusDocument[]> {
    return apiFetch(`${BASE}/corpora/${corpusId}/documents/metadata/bulk`, {
        method: "POST",
        body: JSON.stringify(payload),
    });
}

export async function importDocumentMetadataCsv(
    corpusId: string,
    file: File
): Promise<{ updated: number; errors: string[]; rows_processed: number }> {
    const formData = new FormData();
    formData.append("file", file);
    return apiFetch(`${BASE}/corpora/${corpusId}/documents/metadata/import`, {
        method: "POST",
        body: formData,
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

export type LabelPayload = {
    name?: string;
    description?: string | null;
    inclusion_criteria?: string | null;
    exclusion_criteria?: string | null;
    positive_examples?: string[] | null;
    negative_examples?: string[] | null;
    is_placeholder?: boolean;
};

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

export async function getCodebook(codebookId: string): Promise<Codebook> {
    return apiFetch(`${BASE}/codebooks/${codebookId}`);
}

export async function freezeCodebook(codebookId: string): Promise<Codebook> {
    return apiFetch(`${BASE}/codebooks/${codebookId}/freeze`, { method: "POST" });
}

export async function createCodebookVersion(
    codebookId: string,
    newVersion: string
): Promise<Codebook> {
    return apiFetch(
        `${BASE}/codebooks/${codebookId}/versions?new_version=${encodeURIComponent(newVersion)}`,
        { method: "POST" }
    );
}

export async function listLabels(codebookId: string): Promise<AnnotationLabel[]> {
    return apiFetch(`${BASE}/codebooks/${codebookId}/labels`);
}

export async function addLabel(
    codebookId: string,
    payload: LabelPayload & { name: string }
): Promise<AnnotationLabel> {
    return apiFetch(`${BASE}/codebooks/${codebookId}/labels`, {
        method: "POST",
        body: JSON.stringify(payload),
    });
}

export async function updateLabel(
    labelId: string,
    payload: LabelPayload
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

export async function assignCorpusAnnotationTasks(
    corpusId: string,
    payload: {
        unit_type: UnitType;
        sample_size?: number;
        limit?: number;
        annotator_ids?: string[];
        strategy?: "shared" | "disjoint" | "overlap";
        overlap_count?: number;
        overlap_percent?: number;
    }
): Promise<{
    assigned_count: number;
    unique_units: number;
    overlap_units: number;
    strategy: string;
    unit_type: string;
    per_annotator: Record<string, number>;
}> {
    return apiFetch(`${BASE}/corpora/${corpusId}/annotations/assign`, {
        method: "POST",
        body: JSON.stringify(payload),
    });
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

export async function getTextUnitContext(
    textUnitId: string,
    window = 2
): Promise<{
    unit: {
        id: string;
        corpus_document_id: string;
        unit_type: string;
        position: number;
        text: string;
    };
    document: {
        id: string;
        title: string | null;
        organization: string | null;
        publication_year: number | null;
        country: string | null;
        language: string | null;
    } | null;
    before: Array<{ id: string; position: number; text: string }>;
    after: Array<{ id: string; position: number; text: string }>;
}> {
    return apiFetch(`${BASE}/text-units/${textUnitId}/context?window=${window}`);
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

export type DisagreementItem = {
    text_unit_id: string;
    label_id: string;
    label_name: string;
    judgements: Record<string, string>;
};

export type AdjudicationRecord = {
    id: string;
    text_unit_id: string;
    label_id: string;
    codebook_version: string | null;
    final_value: string;
    adjudicator_id: string;
    comment: string | null;
    created_at: string;
};

export async function listDisagreements(
    corpusId: string,
    codebookId: string
): Promise<DisagreementItem[]> {
    return apiFetch(
        `${BASE}/corpora/${corpusId}/adjudication/disagreements?codebook_id=${encodeURIComponent(codebookId)}`
    );
}

export async function saveAdjudication(payload: {
    text_unit_id: string;
    label_id: string;
    final_value: string;
    comment?: string;
}): Promise<{ id: string }> {
    return apiFetch(`${BASE}/adjudication`, {
        method: "POST",
        body: JSON.stringify(payload),
    });
}

export async function listAdjudications(corpusId: string): Promise<AdjudicationRecord[]> {
    return apiFetch(`${BASE}/corpora/${corpusId}/adjudications`);
}

// ------------------------------------------------------------------
// Analysis
// ------------------------------------------------------------------

export type CorpusFilterParams = {
    organization?: string;
    organization_type?: string;
    publication_year_min?: number;
    publication_year_max?: number;
    region?: string;
    cultural_sphere?: string;
    language?: string;
    publication_type?: string;
};

export type AnalysisBasePayload = {
    unit_type: UnitType;
    preprocessing_profile_id?: string;
} & CorpusFilterParams;

export async function runCorpusStats(
    corpusId: string,
    payload: AnalysisBasePayload
): Promise<AnalysisRun> {
    return apiFetch(`${BASE}/corpora/${corpusId}/analysis/corpus-stats`, {
        method: "POST",
        body: JSON.stringify(payload),
    });
}

export async function runFrequencies(
    corpusId: string,
    payload: AnalysisBasePayload & { top_n?: number }
): Promise<AnalysisRun> {
    return apiFetch(`${BASE}/corpora/${corpusId}/analysis/frequencies`, {
        method: "POST",
        body: JSON.stringify({ top_n: 50, ...payload }),
    });
}

export async function runNgrams(
    corpusId: string,
    payload: AnalysisBasePayload & { n?: number; top_n?: number }
): Promise<AnalysisRun> {
    return apiFetch(`${BASE}/corpora/${corpusId}/analysis/ngrams`, {
        method: "POST",
        body: JSON.stringify({ n: 2, top_n: 50, ...payload }),
    });
}

export async function runDfm(
    corpusId: string,
    payload: AnalysisBasePayload & { weighting?: "count" | "binary" | "tfidf" }
): Promise<AnalysisRun> {
    return apiFetch(`${BASE}/corpora/${corpusId}/analysis/dfm`, {
        method: "POST",
        body: JSON.stringify({ weighting: "count", ...payload }),
    });
}

export async function runKwic(
    corpusId: string,
    payload: AnalysisBasePayload & {
        keyword: string;
        window_size?: number;
        case_sensitive?: boolean;
    }
): Promise<AnalysisRun> {
    return apiFetch(`${BASE}/corpora/${corpusId}/analysis/kwic`, {
        method: "POST",
        body: JSON.stringify({ window_size: 5, case_sensitive: false, ...payload }),
    });
}

export async function runDictionaryAnalysis(
    corpusId: string,
    payload: AnalysisBasePayload & {
        dictionary_id?: string;
        dictionary_terms?: string[];
        group_by?: string;
    }
): Promise<AnalysisRun> {
    return apiFetch(`${BASE}/corpora/${corpusId}/analysis/dictionary`, {
        method: "POST",
        body: JSON.stringify(payload),
    });
}

export async function runKeyness(
    corpusId: string,
    payload: {
        unit_type: UnitType;
        preprocessing_profile_id?: string;
        filters_a: Record<string, unknown>;
        filters_b: Record<string, unknown>;
        top_n?: number;
    }
): Promise<AnalysisRun> {
    return apiFetch(`${BASE}/corpora/${corpusId}/analysis/keyness`, {
        method: "POST",
        body: JSON.stringify({ top_n: 50, ...payload }),
    });
}

export async function runCooccurrence(
    corpusId: string,
    payload: AnalysisBasePayload & { window_size?: number; top_n?: number }
): Promise<AnalysisRun> {
    return apiFetch(`${BASE}/corpora/${corpusId}/analysis/cooccurrence`, {
        method: "POST",
        body: JSON.stringify({ window_size: 5, top_n: 50, ...payload }),
    });
}

export type ResearchDictionary = {
    id: string;
    project_id: string;
    name: string;
    version: string;
    description: string | null;
    terms: string[];
    created_by: string;
    created_at: string;
};

export async function listDictionaries(projectId: string): Promise<ResearchDictionary[]> {
    return apiFetch(`${BASE}/projects/${projectId}/dictionaries`);
}

export async function createDictionary(
    projectId: string,
    payload: { name: string; description?: string; terms: string[] }
): Promise<ResearchDictionary> {
    return apiFetch(`${BASE}/projects/${projectId}/dictionaries`, {
        method: "POST",
        body: JSON.stringify(payload),
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
    selected_annotator_id?: string | null;
    minimum_agreement?: number | null;
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
    preprocessing_profile_id?: string;
    ngram_max?: number;
    min_df?: number;
    max_df?: number;
    max_features?: number | null;
    class_weight?: string | null;
    regularization_c?: number;
    test_size?: number;
    random_seed?: number;
}): Promise<AnalysisRun> {
    return apiFetch(`${BASE}/classifiers/train`, {
        method: "POST",
        body: JSON.stringify({
            algorithm: "logistic_regression",
            run_async: true,
            ...payload,
        }),
    });
}

export type ClassifierCoefficient = {
    feature: string;
    label: string;
    coefficient: number;
    direction: string;
    rank: number;
};

export async function getClassifierCoefficients(
    modelId: string
): Promise<ClassifierCoefficient[]> {
    return apiFetch(`${BASE}/classifiers/${modelId}/coefficients`);
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
    annotation_source?: string;
    selected_annotator_id?: string | null;
    minimum_agreement?: number | null;
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
        algorithm?: "lda" | "nmf" | string;
        n_topics?: number;
        max_iterations?: number;
        random_seed?: number;
        preprocessing_profile_id?: string;
        run_async?: boolean;
        organization?: string;
        region?: string;
        cultural_sphere?: string;
        language?: string;
        publication_year_min?: number;
        publication_year_max?: number;
    }
): Promise<AnalysisRun> {
    return apiFetch(`${BASE}/corpora/${corpusId}/topics/train`, {
        method: "POST",
        body: JSON.stringify({
            algorithm: "lda",
            n_topics: 5,
            run_async: true,
            ...payload,
        }),
    });
}

export async function listTopicLabels(
    runId: string
): Promise<Array<{ id: string; topic_id: number | string; human_name: string; created_at: string }>> {
    return apiFetch(`${BASE}/topics/${runId}/labels`);
}

export async function nameTopic(
    runId: string,
    payload: { topic_id: number | string; human_name: string }
): Promise<{ id: string; topic_id: number | string; human_name: string }> {
    return apiFetch(`${BASE}/topics/${runId}/labels`, {
        method: "POST",
        body: JSON.stringify(payload),
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
        body: JSON.stringify({ algorithm: "logistic_regression", run_async: true, ...payload }),
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
        model_id?: string;
        organization?: string;
        organization_type?: string;
        publication_year_min?: number;
        publication_year_max?: number;
        region?: string;
        cultural_sphere?: string;
        language?: string;
        publication_type?: string;
        country?: string;
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

export type ClonedRunParameters = {
    run_type: string;
    corpus_id: string | null;
    parameters: Record<string, unknown>;
};

export async function cloneRunParameters(runId: string): Promise<ClonedRunParameters> {
    return apiFetch(`${BASE}/runs/${runId}/clone-parameters`);
}

export async function rerunRun(runId: string, run_async = true): Promise<AnalysisRun> {
    return apiFetch(`${BASE}/runs/${runId}/rerun?run_async=${run_async}`, {
        method: "POST",
    });
}

export async function cancelRun(runId: string): Promise<AnalysisRun> {
    return apiFetch(`${BASE}/runs/${runId}/cancel`, { method: "POST" });
}

export type RunCompareDiffRow = {
    parameter?: string;
    metric?: string;
    run_a: unknown;
    run_b: unknown;
    changed: boolean;
};

export type RunCompareResult = {
    run_a: {
        id: string;
        run_type: string;
        status: string;
        created_at: string | null;
        artifact_path: string | null;
        random_seed: number | null;
    };
    run_b: {
        id: string;
        run_type: string;
        status: string;
        created_at: string | null;
        artifact_path: string | null;
        random_seed: number | null;
    };
    parameter_diff: RunCompareDiffRow[];
    metric_diff: RunCompareDiffRow[];
    changed_parameters: RunCompareDiffRow[];
    changed_metrics: RunCompareDiffRow[];
};

export async function compareRuns(runAId: string, runBId: string): Promise<RunCompareResult> {
    return apiFetch(`${BASE}/runs/${runAId}/compare/${runBId}`);
}

export async function exportRunJson(runId: string): Promise<unknown> {
    return apiFetch(`${BASE}/runs/${runId}/export.json`);
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

export type PreprocessingConfigPayload = {
    lowercase?: boolean;
    remove_punctuation?: boolean;
    remove_numbers?: boolean;
    remove_stopwords?: boolean;
    preserve_negation?: boolean;
    stemming?: boolean;
    lemmatization?: boolean;
    ngram_min?: number;
    ngram_max?: number;
    min_df?: number;
    max_df?: number;
    max_features?: number | null;
    custom_stopwords?: string[];
};

export type PreprocessingPreview = {
    rows: Array<{
        original: string;
        processed: string;
        token_count_before: number;
        token_count_after: number;
    }>;
    token_count_before: number;
    token_count_after: number;
    vocabulary_size: number;
    most_frequently_removed_terms: Array<{ term: string; count: number }>;
    config: Record<string, unknown>;
    stemmer: string;
    lemmatization_supported: boolean;
    profile_name: string | null;
    profile_updated_at: string | null;
};

export async function createPreprocessingProfile(
    projectId: string,
    payload: { name: string; description?: string; config: PreprocessingConfigPayload }
): Promise<PreprocessingProfile> {
    return apiFetch(`${BASE}/projects/${projectId}/preprocessing-profiles`, {
        method: "POST",
        body: JSON.stringify(payload),
    });
}

export async function updatePreprocessingProfile(
    profileId: string,
    payload: { name?: string; description?: string; config?: PreprocessingConfigPayload }
): Promise<PreprocessingProfile> {
    return apiFetch(`${BASE}/preprocessing-profiles/${profileId}`, {
        method: "PATCH",
        body: JSON.stringify(payload),
    });
}

export async function deletePreprocessingProfile(profileId: string): Promise<void> {
    await apiFetch(`${BASE}/preprocessing-profiles/${profileId}`, { method: "DELETE" });
}

export async function previewPreprocessing(payload: {
    project_id?: string;
    corpus_id?: string;
    unit_type?: UnitType;
    texts?: string[];
    config?: PreprocessingConfigPayload;
    preprocessing_profile_id?: string;
    sample_size?: number;
}): Promise<PreprocessingPreview> {
    return apiFetch(`${BASE}/preprocessing/preview`, {
        method: "POST",
        body: JSON.stringify(payload),
    });
}

export function researchExportUrl(path: string): string {
    const apiBase = import.meta.env.VITE_API_BASE ?? "/api/v1";
    return `${apiBase}${path}`;
}

// ------------------------------------------------------------------
// Contextual / mixed-method datasets
// ------------------------------------------------------------------

export type ContextualDatasetSummary = {
    id: string;
    project_id: string;
    name: string;
    description: string | null;
    created_by: string;
    created_at: string;
    observation_count: number;
};

export type ContextualDatasetDetail = ContextualDatasetSummary & {
    indicator_keys: string[];
    observations: Array<{
        id: string;
        country: string | null;
        year: number | null;
        values: Record<string, number>;
        created_at: string;
    }>;
};

export type ContextualLinkResult = {
    exploratory: boolean;
    disclaimer: string;
    dataset: { id: string; name: string };
    indicator_key: string;
    group_by: string;
    prevalence_run_id: string;
    point_count: number;
    unmatched_groups: number;
    points: Array<{
        label: string;
        group: string;
        group_by: string;
        prevalence: number;
        yes: number;
        total: number;
        indicator_key: string;
        indicator_value: number;
    }>;
    correlations: Record<
        string,
        { n: number; pearson_r: number | null; status: string }
    >;
};

export async function listContextualDatasets(
    projectId: string
): Promise<ContextualDatasetSummary[]> {
    return apiFetch(`${BASE}/projects/${projectId}/contextual-datasets`);
}

export async function createContextualDataset(
    projectId: string,
    payload: { name: string; description?: string }
): Promise<ContextualDatasetSummary> {
    return apiFetch(`${BASE}/projects/${projectId}/contextual-datasets`, {
        method: "POST",
        body: JSON.stringify(payload),
    });
}

export async function getContextualDataset(
    datasetId: string
): Promise<ContextualDatasetDetail> {
    return apiFetch(`${BASE}/contextual-datasets/${datasetId}`);
}

export async function deleteContextualDataset(datasetId: string): Promise<void> {
    await apiFetch(`${BASE}/contextual-datasets/${datasetId}`, { method: "DELETE" });
}

export async function importContextualCsv(
    datasetId: string,
    file: File,
    replaceExisting = true
): Promise<{
    dataset_id: string;
    imported: number;
    skipped: number;
    indicator_keys: string[];
    replaced: boolean;
}> {
    const formData = new FormData();
    formData.append("file", file);
    const qs = replaceExisting ? "" : "?replace_existing=false";
    return apiFetch(`${BASE}/contextual-datasets/${datasetId}/import-csv${qs}`, {
        method: "POST",
        body: formData,
    });
}

export async function linkContextualDiscourse(
    datasetId: string,
    payload: {
        corpus_id: string;
        codebook_id: string;
        label_ids: string[];
        indicator_key: string;
        unit_type?: UnitType;
        group_by?: "country" | "publication_year" | string;
        provenance_mode?: string;
        model_id?: string;
    }
): Promise<ContextualLinkResult> {
    return apiFetch(`${BASE}/contextual-datasets/${datasetId}/link-discourse`, {
        method: "POST",
        body: JSON.stringify(payload),
    });
}
