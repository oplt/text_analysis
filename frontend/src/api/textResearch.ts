import { apiFetch, apiFetchStream, type Paginated } from "./client";
import type {
    AnalysisRun,
    Annotation,
    AnnotationBlindPolicy,
    AnnotationCampaign,
    AnnotationLabel,
    AnnotationMode,
    AnnotationProgress,
    AnnotationQueueItem,
    Codebook,
    CorpusDocument,
    CleaningPreview,
    CleaningProfile,
    DashboardSummary,
    DatasetPreview,
    ExportManifest,
    PreprocessingProfile,
    ResearchCorpus,
    ResearchMemo,
    TrainedModel,
    TrainingDatasetSnapshot,
    IngestionQaDocument,
    UnitType,
    UncertainPrediction,
} from "../features/text-research/types";

const BASE = "/research";

export type AnalysisCapabilities = {
    operations: Record<string, { async: boolean }>;
    run_types: Record<string, { async: boolean }>;
};

export async function getAnalysisCapabilities(
    signal?: AbortSignal
): Promise<AnalysisCapabilities> {
    return apiFetch(`${BASE}/analysis-capabilities`, { signal });
}

export async function runIngestionQa(corpusId: string): Promise<AnalysisRun> {
    return apiFetch(`${BASE}/corpora/${corpusId}/ingestion-qa`, { method: "POST" });
}

export async function getDocumentIngestionQa(
    documentId: string,
    runId?: string
): Promise<IngestionQaDocument & { run_id: string }> {
    const query = runId ? `?run_id=${encodeURIComponent(runId)}` : "";
    return apiFetch(`${BASE}/documents/${documentId}/ingestion-qa${query}`);
}

export async function listCleaningProfiles(projectId: string): Promise<CleaningProfile[]> {
    return apiFetch(`${BASE}/projects/${projectId}/cleaning-profiles`);
}

export async function createCleaningProfile(
    projectId: string,
    payload: { name: string; description?: string; version?: string; config: Record<string, unknown> }
): Promise<CleaningProfile> {
    return apiFetch(`${BASE}/projects/${projectId}/cleaning-profiles`, {
        method: "POST",
        body: JSON.stringify(payload),
    });
}

export async function updateCleaningProfile(
    profileId: string,
    payload: Partial<{ name: string; description: string; version: string; config: Record<string, unknown> }>
): Promise<CleaningProfile> {
    return apiFetch(`${BASE}/cleaning-profiles/${profileId}`, {
        method: "PATCH",
        body: JSON.stringify(payload),
    });
}

export async function deleteCleaningProfile(profileId: string): Promise<void> {
    await apiFetch(`${BASE}/cleaning-profiles/${profileId}`, { method: "DELETE" });
}

export async function previewCleaning(payload: {
    project_id?: string;
    texts?: string[];
    document_id?: string;
    config?: Record<string, unknown>;
    cleaning_profile_id?: string;
}): Promise<CleaningPreview> {
    return apiFetch(`${BASE}/cleaning/preview`, { method: "POST", body: JSON.stringify(payload) });
}

export async function applyCleaning(
    corpusId: string,
    payload: { cleaning_profile_id: string; document_ids?: string[] }
): Promise<AnalysisRun> {
    return apiFetch(`${BASE}/corpora/${corpusId}/clean`, {
        method: "POST",
        body: JSON.stringify(payload),
    });
}

type StandardAnalysisPayload = {
    unit_type: UnitType;
    preprocessing_profile_id?: string;
    [key: string]: unknown;
};

export async function runSimilarity(
    corpusId: string,
    payload: StandardAnalysisPayload & {
        method?: "tfidf_cosine" | "jaccard";
        mode?: "pairwise" | "query" | "group_centroid";
        top_k?: number;
        min_score?: number;
        group_by?: string;
        query_text?: string;
    }
): Promise<AnalysisRun> {
    return apiFetch(`${BASE}/corpora/${corpusId}/analysis/similarity`, { method: "POST", body: JSON.stringify(payload) });
}

export async function runDuplicateDetection(
    corpusId: string,
    payload: StandardAnalysisPayload & {
        methods?: string[];
        lexical_threshold?: number;
        char_ngram_size?: number;
        use_minhash?: boolean;
        max_pairs?: number;
    }
): Promise<AnalysisRun> {
    return apiFetch(`${BASE}/corpora/${corpusId}/analysis/duplicate-detection`, { method: "POST", body: JSON.stringify(payload) });
}

export async function runClustering(
    corpusId: string,
    payload: StandardAnalysisPayload & {
        n_clusters?: number;
        algorithm?: "kmeans" | "minibatch_kmeans";
        use_svd?: boolean;
        n_svd_components?: number;
        top_terms?: number;
        random_seed?: number;
    }
): Promise<AnalysisRun> {
    return apiFetch(`${BASE}/corpora/${corpusId}/analysis/clustering`, { method: "POST", body: JSON.stringify(payload) });
}

export async function runDimensionalityReduction(
    corpusId: string,
    payload: StandardAnalysisPayload & { method?: "svd" | "pca"; n_components?: number; random_seed?: number }
): Promise<AnalysisRun> {
    return apiFetch(`${BASE}/corpora/${corpusId}/analysis/dimensionality-reduction`, { method: "POST", body: JSON.stringify(payload) });
}

export async function runReadability(
    corpusId: string,
    payload: StandardAnalysisPayload
): Promise<AnalysisRun> {
    return apiFetch(`${BASE}/corpora/${corpusId}/analysis/readability`, { method: "POST", body: JSON.stringify(payload) });
}

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

export async function listCorpora(
    projectId: string,
    signal?: AbortSignal
): Promise<ResearchCorpus[]> {
    return apiFetch(`${BASE}/projects/${projectId}/corpora`, { signal });
}

export async function getCorpus(
    corpusId: string,
    signal?: AbortSignal
): Promise<ResearchCorpus> {
    return apiFetch(`${BASE}/corpora/${corpusId}`, { signal });
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
    },
    signal?: AbortSignal
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
    return apiFetch(`${BASE}/corpora/${corpusId}/documents${qs ? `?${qs}` : ""}`, { signal });
}

export async function getDocument(
    documentId: string,
    signal?: AbortSignal
): Promise<CorpusDocument> {
    return apiFetch(`${BASE}/documents/${documentId}`, { signal });
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

export async function getSourceText(documentId: string): Promise<{
    document_id: string;
    text: string;
    page_provenance?: Array<{
        page_number?: number | null;
        char_start?: number | null;
        char_end?: number | null;
        source_span_ids?: string[] | null;
        offset_coordinate_system?: string | null;
        offset_scope?: "parsed_document" | "page" | "canonical_document" | null;
    }>;
}> {
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
            seed_demo_labels: payload.seed_demo_labels ?? false,
        }),
    });
}

export async function listCodebooks(
    projectId: string,
    signal?: AbortSignal
): Promise<Codebook[]> {
    return apiFetch(`${BASE}/projects/${projectId}/codebooks`, { signal });
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

export async function listAnnotationQueue(
    status?: string,
    params: { limit?: number; offset?: number } = {},
    signal?: AbortSignal
): Promise<Paginated<AnnotationQueueItem>> {
    const search = new URLSearchParams();
    if (status) search.set("status", status);
    if (params.limit !== undefined) search.set("limit", String(params.limit));
    if (params.offset !== undefined) search.set("offset", String(params.offset));
    return apiFetch(`${BASE}/annotations/queue${search.size ? `?${search}` : ""}`, { signal });
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
        // Stratified annotation sampling (prompt.txt §24). Optional; with no
        // stratify_by, sampling is a seeded random draw of sample_size units.
        random_seed?: number;
        stratify_by?: string[];
        stratum_mode?: "proportional" | "equal";
        sampling_level?: "unit" | "document";
        max_units_per_document?: number;
        create_campaign?: boolean;
        campaign_name?: string;
        campaign_description?: string;
        campaign_id?: string;
        codebook_id?: string;
        annotation_mode?: AnnotationMode;
        blind_mode?: boolean;
        ai_assistance_enabled?: boolean;
        reveal_after?: string;
    }
): Promise<{
    assigned_count: number;
    unique_units: number;
    overlap_units: number;
    strategy: string;
    unit_type: string;
    per_annotator: Record<string, number>;
    sampling_plan?: Record<string, unknown> | null;
    campaign_id?: string | null;
    annotation_mode?: AnnotationMode | null;
    blind_mode?: boolean | null;
    ai_assistance_enabled?: boolean | null;
}> {
    return apiFetch(`${BASE}/corpora/${corpusId}/annotations/assign`, {
        method: "POST",
        body: JSON.stringify(payload),
    });
}

export async function listAnnotationCampaigns(
    projectId: string,
    corpusId?: string,
    signal?: AbortSignal
): Promise<AnnotationCampaign[]> {
    const search = new URLSearchParams();
    if (corpusId) search.set("corpus_id", corpusId);
    const qs = search.size ? `?${search}` : "";
    return apiFetch(`${BASE}/projects/${projectId}/annotation-campaigns${qs}`, { signal });
}

export async function createAnnotationCampaign(
    projectId: string,
    payload: {
        corpus_id: string;
        name: string;
        description?: string;
        codebook_id?: string;
        unit_type?: UnitType;
        sampling_strategy?: string;
        assignment_strategy?: string;
        sample_size?: number;
        overlap_count?: number;
        overlap_percent?: number;
        annotation_mode?: AnnotationMode;
        blind_mode?: boolean;
        ai_assistance_enabled?: boolean;
        reveal_after?: string;
        annotator_ids?: string[];
        metadata?: Record<string, unknown>;
    }
): Promise<AnnotationCampaign> {
    return apiFetch(`${BASE}/projects/${projectId}/annotation-campaigns`, {
        method: "POST",
        body: JSON.stringify(payload),
    });
}

export async function getAnnotationCampaign(campaignId: string): Promise<AnnotationCampaign> {
    return apiFetch(`${BASE}/annotation-campaigns/${campaignId}`);
}

export async function getTextUnitBlindPolicy(textUnitId: string): Promise<AnnotationBlindPolicy> {
    return apiFetch(`${BASE}/text-units/${textUnitId}/blind-policy`);
}

export async function saveAnnotations(payload: {
    text_unit_id: string;
    codebook_id: string;
    campaign_id?: string;
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

export async function listUnitAnnotations(
    textUnitId: string,
    signal?: AbortSignal
): Promise<Annotation[]> {
    return apiFetch(`${BASE}/text-units/${textUnitId}/annotations`, { signal });
}

export async function listAnnotationsForUnits(
    corpusId: string,
    textUnitIds: string[],
    campaignId?: string,
    signal?: AbortSignal
): Promise<Annotation[]> {
    if (!textUnitIds.length) return [];
    const search = new URLSearchParams();
    for (const id of textUnitIds) {
        search.append("text_unit_ids", id);
    }
    if (campaignId) search.set("campaign_id", campaignId);
    return apiFetch(`${BASE}/corpora/${encodeURIComponent(corpusId)}/annotations?${search}`, { signal });
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

export type ComputeReliabilityPayload = {
    codebook_id: string;
    label_ids?: string[];
    campaign_id?: string;
    unit_type?: string;
    annotator_ids?: string[];
    bootstrap_samples?: number;
    confidence_level?: number;
    random_seed?: number;
};

export async function computeReliability(
    corpusId: string,
    payload: ComputeReliabilityPayload
): Promise<AnalysisRun> {
    return apiFetch(`${BASE}/corpora/${corpusId}/reliability`, {
        method: "POST",
        body: JSON.stringify(payload),
    });
}

export async function computeCampaignReliability(
    campaignId: string,
    payload: ComputeReliabilityPayload
): Promise<AnalysisRun> {
    return apiFetch(`${BASE}/annotation-campaigns/${campaignId}/reliability`, {
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

export async function listCampaignDisagreements(
    campaignId: string,
    codebookId?: string
): Promise<DisagreementItem[]> {
    const query = codebookId ? `?codebook_id=${encodeURIComponent(codebookId)}` : "";
    return apiFetch(`${BASE}/annotation-campaigns/${campaignId}/disagreements${query}`);
}

export async function saveCampaignAdjudication(
    campaignId: string,
    payload: Omit<Parameters<typeof saveAdjudication>[0], "campaign_id">
): Promise<{ id: string }> {
    return apiFetch(`${BASE}/annotation-campaigns/${campaignId}/adjudications`, {
        method: "POST",
        body: JSON.stringify(payload),
    });
}

export async function listCampaignAdjudications(
    campaignId: string
): Promise<AdjudicationRecord[]> {
    return apiFetch(`${BASE}/annotation-campaigns/${campaignId}/adjudications`);
}

// ------------------------------------------------------------------
// Analysis
// ------------------------------------------------------------------

export type CorpusFilterParams = {
    organization?: string;
    organization_type?: string;
    /** Exact year facet (MetadataFilterBar); also supports min/max range below. */
    publication_year?: number;
    publication_year_min?: number;
    publication_year_max?: number;
    country?: string;
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
    payload: AnalysisBasePayload & {
        weighting?: "count" | "binary" | "tf" | "tfidf" | "sublinear_tf" | "log_count" | "bm25";
        k1?: number;
        b?: number;
        smooth_idf?: boolean;
        force_sparse_only?: boolean;
        trim?: {
            min_term_frequency?: number;
            max_term_frequency?: number;
            term_frequency_type?: "count" | "prop" | "rank" | "quantile";
            min_document_frequency?: number;
            max_document_frequency?: number;
            document_frequency_type?: "count" | "prop" | "rank" | "quantile";
            top_n?: number;
        };
    }
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
        query_mode?: string;
        /** Lemma / linguistic query language (not a corpus metadata filter). */
        query_language?: string;
        /** Alias for query_language. */
        kwic_language?: string;
        token_attribute?: string;
        max_matches?: number;
    }
): Promise<AnalysisRun> {
    return apiFetch(`${BASE}/corpora/${corpusId}/analysis/kwic`, {
        method: "POST",
        body: JSON.stringify({
            window_size: 5,
            case_sensitive: false,
            query_mode: "auto",
            ...payload,
        }),
    });
}

export async function runDictionaryAnalysis(
    corpusId: string,
    payload: AnalysisBasePayload & {
        dictionary_id?: string;
        dictionary_terms?: string[];
        hierarchy?: Record<string, unknown>;
        group_by?: string;
    }
): Promise<AnalysisRun> {
    if (
        !payload.dictionary_id &&
        !payload.dictionary_terms?.length &&
        !Object.keys(payload.hierarchy ?? {}).length
    ) {
        throw new Error("dictionary_id, dictionary_terms, or hierarchy is required.");
    }
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
        group_field?: string;
        method?: string;
        correction?: string;
        min_frequency?: number;
        top_n?: number;
    }
): Promise<AnalysisRun> {
    return apiFetch(`${BASE}/corpora/${corpusId}/analysis/keyness`, {
        method: "POST",
        body: JSON.stringify({
            top_n: 50,
            method: "log_likelihood",
            correction: "bh",
            min_frequency: 1,
            ...payload,
        }),
    });
}

export async function runCooccurrence(
    corpusId: string,
    payload: AnalysisBasePayload & {
        window_size?: number;
        top_n?: number;
        association_method?: string;
        directional?: boolean;
        min_frequency?: number;
        min_count?: number;
    }
): Promise<AnalysisRun> {
    return apiFetch(`${BASE}/corpora/${corpusId}/analysis/cooccurrence`, {
        method: "POST",
        body: JSON.stringify({
            window_size: 5,
            top_n: 50,
            association_method: "pmi",
            directional: false,
            min_frequency: 1,
            min_count: 1,
            ...payload,
        }),
    });
}

export type ResearchDictionary = {
    id: string;
    project_id: string;
    name: string;
    version: string;
    description: string | null;
    language?: string | null;
    terms: string[];
    hierarchy?: Record<string, unknown> | null;
    exclusions?: Array<Record<string, unknown>>;
    format?: string;
    created_by: string;
    created_at: string;
};

export async function listDictionaries(
    projectId: string,
    signal?: AbortSignal
): Promise<ResearchDictionary[]> {
    return apiFetch(`${BASE}/projects/${projectId}/dictionaries`, { signal });
}

export async function createDictionary(
    projectId: string,
    payload: {
        name: string;
        description?: string;
        version?: string;
        language?: string;
        terms?: unknown[];
        hierarchy?: Record<string, unknown>;
        exclusions?: unknown[];
    }
): Promise<ResearchDictionary> {
    return apiFetch(`${BASE}/projects/${projectId}/dictionaries`, {
        method: "POST",
        body: JSON.stringify(payload),
    });
}

export async function getDictionary(dictionaryId: string): Promise<ResearchDictionary> {
    return apiFetch(`${BASE}/dictionaries/${dictionaryId}`);
}

export async function updateDictionary(
    dictionaryId: string,
    payload: {
        name?: string;
        description?: string;
        version?: string;
        language?: string;
        terms?: unknown[];
        hierarchy?: Record<string, unknown>;
        exclusions?: unknown[];
    }
): Promise<ResearchDictionary> {
    return apiFetch(`${BASE}/dictionaries/${dictionaryId}`, {
        method: "PATCH",
        body: JSON.stringify(payload),
    });
}

export async function createDictionaryVersion(dictionaryId: string): Promise<ResearchDictionary> {
    return apiFetch(`${BASE}/dictionaries/${dictionaryId}/versions`, { method: "POST" });
}

export type MetadataFacets = Record<string, Array<{ value: string; count: number }>>;

export async function getCorpusMetadataFacets(
    corpusId: string,
    signal?: AbortSignal
): Promise<MetadataFacets> {
    return apiFetch(`${BASE}/corpora/${corpusId}/facets`, { signal });
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
    annotation_campaign_id?: string | null;
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
    task_type?: "binary" | "multiclass" | "multilabel";
    name?: string;
    run_async?: boolean;
    preprocessing_profile_id?: string;
    vectorizer?: "tfidf" | "count";
    use_word_ngrams?: boolean;
    ngram_min?: number;
    ngram_max?: number;
    use_char_ngrams?: boolean;
    char_ngram_min?: number;
    char_ngram_max?: number;
    min_df?: number;
    max_df?: number;
    max_features?: number | null;
    feature_selection_method?: "none" | "chi2" | "mutual_info" | "l1";
    feature_selection_k?: number | "all";
    feature_selection_percentile?: number | null;
    class_weight?: string | null;
    regularization_c?: number;
    nb_alpha?: number;
    sgd_loss?: string;
    test_size?: number;
    val_size?: number;
    random_seed?: number;
    tune_hyperparameters?: boolean;
    hyperparameter_search_type?: "grid" | "random";
    hyperparameter_param_grid?: Record<string, unknown[]>;
    hyperparameter_n_iter?: number;
    hyperparameter_scoring?: string;
    tune_thresholds?: boolean;
    threshold_objective?: string;
    threshold_utility_tp?: number;
    threshold_utility_tn?: number;
    threshold_utility_fp?: number;
    threshold_utility_fn?: number;
    n_bootstrap?: number;
    ci_confidence_level?: number;
    calibration_method?: "sigmoid" | "isotonic";
    validation_strategy?: "holdout" | "nested_grouped_cv";
    nested_cv_outer_splits?: number;
    nested_cv_inner_splits?: number;
    embedding_provider?: string;
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
    payload: {
        unit_type: UnitType;
        only_unannotated?: boolean;
        filters?: CorpusFilterParams & Record<string, unknown>;
    }
): Promise<AnalysisRun> {
    return apiFetch(`${BASE}/classifiers/${modelId}/predict`, {
        method: "POST",
        body: JSON.stringify(payload),
    });
}

export type ActiveLearningQueuePage = {
    items: UncertainPrediction[];
    total: number;
    limit: number;
    offset: number;
};

export async function listUncertainPredictions(
    modelId: string,
    options: {
        limit?: number;
        offset?: number;
        campaignId?: string;
        textUnitId?: string;
        contentMode?: "snippet" | "full";
    } = {}
): Promise<ActiveLearningQueuePage> {
    const search = new URLSearchParams();
    search.set("limit", String(options.limit ?? 20));
    search.set("offset", String(options.offset ?? 0));
    search.set("content_mode", options.contentMode ?? "snippet");
    if (options.campaignId) search.set("campaign_id", options.campaignId);
    if (options.textUnitId) search.set("text_unit_id", options.textUnitId);
    return apiFetch(`${BASE}/classifiers/${modelId}/active-learning/queue?${search}`);
}

export type ActiveLearningAssignment = {
    id: string;
    text_unit_id: string;
    annotator_id: string;
    status: string;
    assigned_at: string | null;
    completed_at: string | null;
};

export async function assignUncertainPredictions(
    modelId: string,
    textUnitIds: string[],
    annotatorIds: string[]
): Promise<ActiveLearningAssignment[]> {
    return apiFetch(`${BASE}/classifiers/${modelId}/active-learning/assign`, {
        method: "POST",
        body: JSON.stringify({ text_unit_ids: textUnitIds, annotator_ids: annotatorIds }),
    });
}

export async function listClassifiers(
    projectId: string,
    corpusId?: string,
    lifecycleStatus?: string,
    signal?: AbortSignal
): Promise<TrainedModel[]> {
    const search = new URLSearchParams();
    if (corpusId) search.set("corpus_id", corpusId);
    if (lifecycleStatus) search.set("lifecycle_status", lifecycleStatus);
    const qs = search.toString() ? `?${search}` : "";
    return apiFetch(`${BASE}/projects/${projectId}/classifiers${qs}`, { signal });
}

export async function listModels(
    projectId: string,
    options: { corpusId?: string; lifecycleStatus?: string } = {},
    signal?: AbortSignal
): Promise<TrainedModel[]> {
    const search = new URLSearchParams();
    if (options.corpusId) search.set("corpus_id", options.corpusId);
    if (options.lifecycleStatus) search.set("lifecycle_status", options.lifecycleStatus);
    const qs = search.toString() ? `?${search}` : "";
    return apiFetch(`${BASE}/projects/${projectId}/models${qs}`, { signal });
}

export async function getClassifier(modelId: string): Promise<TrainedModel> {
    return apiFetch(`${BASE}/classifiers/${encodeURIComponent(modelId)}`);
}

export async function updateModelLifecycle(
    modelId: string,
    payload: {
        status: "candidate" | "staging" | "production" | "deprecated" | "archived";
        notes?: string | null;
        deprecate_others?: boolean;
    }
): Promise<TrainedModel> {
    return apiFetch(`${BASE}/models/${encodeURIComponent(modelId)}/lifecycle`, {
        method: "PATCH",
        body: JSON.stringify(payload),
    });
}

export type ModelLifecycleEvent = {
    id: string;
    model_id: string;
    from_status: string | null;
    to_status: string;
    actor_id: string | null;
    reason: string | null;
    run_id: string | null;
    metadata: Record<string, unknown>;
    created_at: string;
};

export async function listModelLifecycleEvents(modelId: string): Promise<ModelLifecycleEvent[]> {
    return apiFetch(`${BASE}/models/${encodeURIComponent(modelId)}/lifecycle-events`);
}

export async function cloneClassifierConfig(modelId: string): Promise<Record<string, unknown>> {
    return apiFetch(`${BASE}/classifiers/${encodeURIComponent(modelId)}/clone`, {
        method: "POST",
    });
}

export async function listModelPredictions(
    modelId: string,
    options: { limit?: number; offset?: number } = {},
    signal?: AbortSignal
): Promise<unknown[]> {
    const search = new URLSearchParams();
    search.set("limit", String(options.limit ?? 100));
    search.set("offset", String(options.offset ?? 0));
    return apiFetch(
        `${BASE}/classifiers/${encodeURIComponent(modelId)}/predictions?${search}`,
        { signal }
    );
}

export async function listPredictionSets(
    corpusId: string,
    options: { limit?: number; offset?: number } = {},
    signal?: AbortSignal
): Promise<
    Array<{
        id: string;
        project_id: string;
        corpus_id: string;
        trained_model_id: string;
        model_version: number;
        dataset_snapshot_id: string | null;
        analysis_run_id: string;
        created_by: string;
        created_at: string;
        metadata: Record<string, unknown>;
    }>
> {
    const search = new URLSearchParams();
    search.set("limit", String(options.limit ?? 50));
    search.set("offset", String(options.offset ?? 0));
    return apiFetch(`${BASE}/corpora/${encodeURIComponent(corpusId)}/prediction-sets?${search}`, { signal });
}

export async function getPredictionSet(predictionSetId: string): Promise<{
    id: string;
    project_id: string;
    corpus_id: string;
    trained_model_id: string;
    model_version: number;
    dataset_snapshot_id: string | null;
    analysis_run_id: string;
    created_by: string;
    created_at: string;
    metadata: Record<string, unknown>;
    predictions: Array<{
        id: string;
        trained_model_id: string;
        text_unit_id: string;
        predicted_labels: string[];
        scores: Record<string, number>;
        uncertainty: number | null;
        created_at: string;
    }>;
}> {
    return apiFetch(`${BASE}/prediction-sets/${encodeURIComponent(predictionSetId)}`);
}

export type PredictionSetPredictionPage = {
    items: Array<{
        prediction: {
            id: string;
            trained_model_id: string;
            text_unit_id: string;
            predicted_labels: string[];
            scores: Record<string, number>;
            uncertainty: number | null;
            created_at: string;
        };
        human_annotations: Array<{ label_id: string; value: string }>;
        adjudications: Array<{ label_id: string; final_value: string }>;
        review_status: "unreviewed" | "annotated" | "adjudicated";
        human_disagreement: boolean;
        provenance_layers: Record<string, unknown>;
    }>;
    total: number;
    limit: number;
    offset: number;
};

export async function listPredictionSetPredictions(
    predictionSetId: string,
    options: {
        limit?: number;
        offset?: number;
        predictedLabel?: string;
        minConfidence?: number;
        maxUncertainty?: number;
        reviewStatus?: string;
        humanDisagreement?: boolean;
    } = {},
    signal?: AbortSignal
): Promise<PredictionSetPredictionPage> {
    const query = new URLSearchParams({
        limit: String(options.limit ?? 100),
        offset: String(options.offset ?? 0),
    });
    if (options.predictedLabel) query.set("predicted_label", options.predictedLabel);
    if (options.minConfidence != null) query.set("min_confidence", String(options.minConfidence));
    if (options.maxUncertainty != null) query.set("max_uncertainty", String(options.maxUncertainty));
    if (options.reviewStatus) query.set("review_status", options.reviewStatus);
    if (options.humanDisagreement) query.set("human_disagreement", "true");
    return apiFetch(
        `${BASE}/prediction-sets/${encodeURIComponent(predictionSetId)}/predictions?${query}`,
        { signal }
    );
}

export async function compareClassifierDrift(
    corpusId: string,
    payload: {
        /** Preferred: backend aggregates the full persisted PredictionSet. */
        baseline_prediction_set_id?: string;
        current_prediction_set_id?: string;
        mode?: "DATA_DRIFT" | "PREDICTION_DRIFT" | "PERFORMANCE_DRIFT" | "MODEL_COMPARISON";
        /** Legacy/manual aggregate body (browser-built). Prefer prediction set IDs. */
        baseline?: {
            label_counts?: Record<string, number>;
            scores?: number[];
            top_terms?: string[];
        };
        current?: {
            label_counts?: Record<string, number>;
            scores?: number[];
            top_terms?: string[];
        };
        baseline_run_id?: string | null;
        current_run_id?: string | null;
    }
): Promise<Record<string, unknown>> {
    return apiFetch(`${BASE}/corpora/${encodeURIComponent(corpusId)}/monitoring/drift`, {
        method: "POST",
        body: JSON.stringify(payload),
    });
}

export async function listDatasetSnapshots(
    projectId: string,
    corpusId?: string,
    signal?: AbortSignal
): Promise<TrainingDatasetSnapshot[]> {
    const qs = corpusId ? `?corpus_id=${encodeURIComponent(corpusId)}` : "";
    return apiFetch(`${BASE}/projects/${projectId}/dataset-snapshots${qs}`, { signal });
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
        algorithm?: "lda" | "nmf" | "semantic_stack" | "bertopic" | string;
        n_topics?: number;
        max_iterations?: number;
        random_seed?: number;
        preprocessing_profile_id?: string;
        holdout_fraction?: number;
        holdout_unit_ids?: string[];
        group_by?: string[];
        embedding_provider?: "hashing" | "sentence_transformers" | string;
        embedding_model_name?: string;
        persist_embedding_artifacts?: boolean;
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

export async function runTopicKSweep(
    corpusId: string,
    payload: {
        unit_type: UnitType;
        algorithm?: "lda" | "nmf" | string;
        k_values: number[];
        max_iterations?: number;
        random_seed?: number;
        preprocessing_profile_id?: string;
        holdout_fraction?: number;
        holdout_unit_ids?: string[];
        run_async?: boolean;
        organization?: string;
        region?: string;
        cultural_sphere?: string;
        language?: string;
        publication_year_min?: number;
        publication_year_max?: number;
    }
): Promise<AnalysisRun> {
    return apiFetch(`${BASE}/corpora/${corpusId}/topics/k-sweep`, {
        method: "POST",
        body: JSON.stringify({
            algorithm: "lda",
            run_async: true,
            ...payload,
        }),
    });
}

export async function runTopicSeedStability(
    corpusId: string,
    payload: {
        unit_type: UnitType;
        algorithm?: "lda" | "nmf" | string;
        n_topics?: number;
        seeds: number[];
        max_iterations?: number;
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
    return apiFetch(`${BASE}/corpora/${corpusId}/topics/seed-stability`, {
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
    group_field?: string;
    max_groups?: number;
    temporal_field?: string;
    temporal_windows?: boolean;
    transfer_field?: string;
    transfer_train_values?: string[];
    transfer_test_values?: string[];
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

export async function fitStatisticalModel(
    corpusId: string,
    payload: {
        model?: "ols" | "logistic" | string;
        dependent_var: string;
        independent_vars: string[];
        rows: Array<Record<string, unknown>>;
        add_intercept?: boolean;
    }
): Promise<AnalysisRun> {
    return apiFetch(`${BASE}/corpora/${corpusId}/analysis/statistical-model`, {
        method: "POST",
        body: JSON.stringify({
            model: "ols",
            add_intercept: true,
            ...payload,
        }),
    });
}

export async function compareMeasurements(
    corpusId: string,
    payload: {
        source_a: string;
        values_a: unknown[];
        source_b: string;
        values_b: unknown[];
        ids?: string[];
        value_kind?: "categorical" | "continuous" | string;
        subgroup?: string[];
    }
): Promise<AnalysisRun> {
    return apiFetch(`${BASE}/corpora/${corpusId}/analysis/measurement-comparison`, {
        method: "POST",
        body: JSON.stringify({
            value_kind: "categorical",
            ...payload,
        }),
    });
}

// ------------------------------------------------------------------
// Dashboard & runs
// ------------------------------------------------------------------

export async function getDashboardSummary(
    corpusId: string,
    signal?: AbortSignal
): Promise<DashboardSummary> {
    return apiFetch(`${BASE}/corpora/${corpusId}/dashboard`, { signal });
}

export async function listRuns(
    projectId: string,
    params?: { corpus_id?: string; run_type?: string; limit?: number; offset?: number },
    signal?: AbortSignal
): Promise<Paginated<AnalysisRun>> {
    const search = new URLSearchParams();
    if (params?.corpus_id) search.set("corpus_id", params.corpus_id);
    if (params?.run_type) search.set("run_type", params.run_type);
    if (params?.limit != null) search.set("limit", String(params.limit));
    if (params?.offset != null) search.set("offset", String(params.offset));
    const qs = search.toString();
    return apiFetch(`${BASE}/projects/${projectId}/runs${qs ? `?${qs}` : ""}`, { signal });
}

export async function getRun(runId: string, signal?: AbortSignal): Promise<AnalysisRun> {
    return apiFetch(`${BASE}/runs/${runId}`, { signal });
}

export async function getRunResultArtifact(
    runId: string,
    signal?: AbortSignal
): Promise<Record<string, unknown>> {
    return apiFetch(`${BASE}/runs/${runId}/results-artifact`, { signal });
}

export type ClonedRunParameters = {
    run_type: string;
    corpus_id: string | null;
    parameters: Record<string, unknown>;
    reproduce?: Record<string, unknown>;
    analysis_specification?: Record<string, unknown> | null;
    analysis_spec_hash?: string | null;
};

export async function cloneRunParameters(runId: string): Promise<ClonedRunParameters> {
    return apiFetch(`${BASE}/runs/${runId}/clone-parameters`);
}

export type RunProvenance = {
    run_id: string;
    run_type: string;
    status: string;
    random_seed: number | null;
    artifact_path: string | null;
    created_by?: string | null;
    started_at?: string | null;
    completed_at?: string | null;
    corpus_id?: string | null;
    project_id?: string | null;
    provenance: Record<string, unknown>;
    reproduce: Record<string, unknown>;
    runtime_now: Record<string, unknown>;
};

export async function getRunProvenance(
    runId: string,
    signal?: AbortSignal
): Promise<RunProvenance> {
    return apiFetch(`${BASE}/runs/${runId}/provenance`, { signal });
}

export async function rerunRun(
    runId: string, run_async = true, exact = false
): Promise<AnalysisRun> {
    return apiFetch(`${BASE}/runs/${runId}/rerun?run_async=${run_async}&exact=${exact}`, {
        method: "POST",
    });
}

export type RunResultsPage = {
    artifact_id: string | null;
    checksum: string | null;
    key: string | null;
    items: unknown[] | null;
    data: unknown;
    total: number | null;
    limit: number;
    offset: number;
};

export async function getRunResults(
    runId: string,
    options: { key?: string; limit?: number; offset?: number } = {}
): Promise<RunResultsPage> {
    const query = new URLSearchParams({
        limit: String(options.limit ?? 100),
        offset: String(options.offset ?? 0),
    });
    if (options.key) query.set("key", options.key);
    return apiFetch(`${BASE}/runs/${encodeURIComponent(runId)}/results?${query}`);
}

export function runResultsDownloadUrl(runId: string): string {
    return researchExportUrl(`${BASE}/runs/${encodeURIComponent(runId)}/results/download`);
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

export async function getExportManifest(
    corpusId: string,
    signal?: AbortSignal
): Promise<ExportManifest> {
    return apiFetch(`${BASE}/corpora/${corpusId}/export/manifest`, { signal });
}

export async function getQuantedaScript(corpusId: string): Promise<{ script: string }> {
    return apiFetch(`${BASE}/corpora/${corpusId}/export/quanteda-script`);
}

export async function listPreprocessingProfiles(
    projectId: string,
    signal?: AbortSignal
): Promise<PreprocessingProfile[]> {
    return apiFetch(`${BASE}/projects/${projectId}/preprocessing-profiles`, { signal });
}

export type PreprocessingConfigPayload = {
    language?: string;
    language_mode?: "manual" | "auto" | "per_unit" | string;
    auto_detect_language?: boolean;
    multilingual?: boolean;
    unicode_normalization?: string | null;
    fix_encoding?: boolean;
    lowercase?: boolean;
    remove_punctuation?: boolean;
    remove_numbers?: boolean;
    remove_stopwords?: boolean;
    preserve_negation?: boolean;
    stemming?: boolean;
    lemmatization?: boolean;
    pos_lemmatization?: boolean;
    spacy_model?: string;
    enable_ner?: boolean;
    entity_masking?: boolean;
    phrase_detection?: boolean;
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
    spacy_available?: boolean;
    implementation?: Record<string, unknown>;
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

export function researchRunEventsUrl(runId: string): string {
    return researchExportUrl(`${BASE}/runs/${encodeURIComponent(runId)}/events`);
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
};

export type ContextualObservation = {
    id: string;
    country: string | null;
    year: number | null;
    values: Record<string, number>;
    created_at: string;
};

export type ContextualObservationPage = {
    items: ContextualObservation[];
    total: number;
    limit: number;
    offset: number;
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

export async function listContextualObservations(
    datasetId: string,
    params: { limit: number; offset: number }
): Promise<ContextualObservationPage> {
    const search = new URLSearchParams({
        limit: String(params.limit),
        offset: String(params.offset),
    });
    return apiFetch(`${BASE}/contextual-datasets/${datasetId}/observations?${search}`);
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

/* --- Ask Corpus / Research Assistant --- */

export type AssistantScope = {
    corpus_id: string;
    project_id: string;
    corpus_name: string;
    rag_document_ids: string[];
    corpus_document_ids: string[];
    indexed_rag_document_ids: string[];
    unavailable_rag_document_ids: string[];
    scope_hash: string;
    evidence_revision_hash?: string | null;
    index_version: string | null;
    retrieval_version: string | null;
    total_documents: number;
    indexed_count: number;
    unavailable_count: number;
    unavailable_corpus_document_ids: string[];
    unavailable_reasons: Record<string, string>;
    /** Canonical corpus/RAG identity mapping (document_bindings is legacy). */
    documents: CorpusScopeDocumentBinding[];
    document_bindings: CorpusScopeDocumentBinding[];
    warnings: string[];
    scope_mode: "fixed" | "live";
    mapping_status?: "ok" | "unverifiable";
};

export type CorpusScopeDocumentBinding = {
    corpus_document_id: string;
    rag_document_id: string | null;
    availability: "indexed" | "unavailable";
    status: string | null;
    unavailable_reason: string | null;
    index_revision_id: string | null;
    document_revision: string | null;
};

export type AssistantCitation = {
    document_id: string;
    chunk_id: string;
    filename: string;
    score: number;
    snippet: string;
    page_number: number | null;
    chunk_index: number | null;
    citation_number: number | null;
    used_in_answer: boolean;
    section_heading: string | null;
    corpus_document_id?: string | null;
    char_start?: number | null;
    char_end?: number | null;
    source_span_ids?: string[] | null;
    offset_coordinate_system?: string | null;
    offset_scope?: "parsed_document" | "page" | "canonical_document" | null;
    offset_scope_id?: string | null;
    source_spans?: Array<Record<string, unknown>> | null;
    parent_context_id?: string | null;
    index_revision_id?: string | null;
    source_status?: string | null;
};

export type AssistantClaim = {
    text: string;
    chunk_ids: string[];
    citation_numbers: number[];
};

export type AssistantCoverage = {
    documents_in_scope: number;
    documents_with_retrieved_evidence: number;
    retrieved_passage_count: number;
    coverage_ratio: number;
};

export type AssistantSynthesisProvenance = {
    schema_version?: string;
    original_research_question?: string;
    evidence_revision_hash?: string | null;
    corpus_id?: string;
    project_id?: string;
    created_by?: string;
    created_at?: string;
    documents_in_scope?: string[];
    documents_considered?: string[];
    documents_omitted?: Array<{ rag_document_id: string; reason: string }>;
    retrieval_trace_ids?: string[];
    citation_validation_status?: string | null;
};

export type AssistantMessageResult = {
    thread_id: string;
    conversation_id: string;
    message_id: string;
    retrieval_trace_id: string | null;
    query: string;
    original_query?: string | null;
    resolved_retrieval_query?: string | null;
    answer: string;
    citations: AssistantCitation[];
    claims: AssistantClaim[];
    retrieved_chunk_ids: string[];
    model_name: string;
    latency_ms: number;
    no_context_found: boolean;
    retrieval_degraded: boolean;
    degradation_reason: string | null;
    citation_validation_failed: boolean;
    citation_validation_status?: string;
    evidence_revision_hash?: string | null;
    injection_chunks_filtered: number;
    scope: AssistantScope;
    coverage: AssistantCoverage;
    context_message_ids?: string[];
    ai_run_id?: string | null;
    fusion_method?: string | null;
    synthesis_provenance?: AssistantSynthesisProvenance;
};

export type AssistantSynthesizeResult =
    | {
          mode: "sync";
          answer: string;
          citations?: AssistantCitation[];
          claims?: AssistantClaim[];
          scope: AssistantScope;
          document_findings?: unknown[];
          retrieval_trace_ids?: string[];
          documents_total?: number;
          documents_considered?: number;
          documents_with_evidence?: number;
          truncated?: boolean;
          coverage: AssistantCoverage;
          retrieval_trace_id?: string | null;
          evidence_revision_hash?: string | null;
          citation_validation_status?: string;
          synthesis_provenance?: AssistantSynthesisProvenance;
      }
    | {
          mode: "async";
          run_id: string;
          status: string;
          scope: AssistantScope;
          message?: string;
      };

export type AssistantThread = {
    id: string;
    corpus_id: string;
    project_id: string;
    rag_conversation_id: string;
    title: string;
    created_at: string;
    updated_at: string;
    scope_mode: "fixed" | "live";
};

export async function getAssistantScope(corpusId: string): Promise<AssistantScope> {
    return apiFetch(`${BASE}/corpora/${corpusId}/assistant/scope`);
}

export async function listAssistantThreads(corpusId: string): Promise<AssistantThread[]> {
    return apiFetch(`${BASE}/corpora/${corpusId}/assistant/conversations`);
}

export async function createAssistantThread(
    corpusId: string,
    payload?: { title?: string; document_ids?: string[]; scope_mode?: "fixed" | "live" }
): Promise<AssistantThread> {
    return apiFetch(`${BASE}/corpora/${corpusId}/assistant/conversations`, {
        method: "POST",
        body: JSON.stringify(payload ?? {}),
    });
}

export async function getAssistantConversation(threadId: string): Promise<{
    thread: AssistantThread;
    messages: Array<Record<string, unknown>>;
    scope: AssistantScope | null;
    scope_events: AssistantScopeEvent[];
}> {
    return apiFetch(`${BASE}/assistant/conversations/${threadId}`);
}

export type AssistantScopeEvent = {
    id: string;
    thread_id: string;
    actor_id: string | null;
    action: string;
    scope_mode: "fixed" | "live";
    corpus_id: string;
    project_id: string;
    previous_scope_hash: string | null;
    new_scope_hash: string;
    evidence_revision_hash?: string | null;
    rag_document_ids: string[];
    corpus_document_ids: string[];
    reason: string | null;
    created_at: string;
};

export async function updateAssistantThreadScope(
    threadId: string,
    payload: {
        document_ids?: string[] | null;
        scope_mode?: "fixed" | "live" | null;
        reason?: string | null;
    }
): Promise<AssistantScope> {
    return apiFetch(`${BASE}/assistant/conversations/${threadId}/scope`, {
        method: "PATCH",
        body: JSON.stringify(payload),
    });
}

export async function postAssistantMessage(
    corpusId: string,
    payload: {
        query: string;
        thread_id?: string | null;
        intent?: string | null;
        document_ids?: string[] | null;
    }
): Promise<AssistantMessageResult> {
    return apiFetch(`${BASE}/corpora/${corpusId}/assistant/messages`, {
        method: "POST",
        body: JSON.stringify(payload),
    });
}

export type AssistantMessageStreamEvent =
    | "turn_created"
    | "retrieval_started"
    | "retrieval_complete"
    | "generation_started"
    | "citation_validation_complete";

export async function postAssistantMessageStream(
    corpusId: string,
    payload: {
        query: string;
        thread_id?: string | null;
        intent?: string | null;
        document_ids?: string[] | null;
    },
    onEvent: (event: AssistantMessageStreamEvent, payload: Record<string, unknown>) => void,
    signal?: AbortSignal
): Promise<AssistantMessageResult> {
    const response = await apiFetchStream(`${BASE}/corpora/${corpusId}/assistant/messages/stream`, {
        method: "POST",
        body: JSON.stringify(payload),
        signal,
    });
    if (!response.body) throw new Error("Assistant stream was unavailable.");

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
        const { done, value } = await reader.read();
        buffer += decoder.decode(value, { stream: !done });
        const events = buffer.split("\n\n");
        buffer = events.pop() ?? "";
        for (const item of events) {
            const event = item.match(/^event: (.+)$/m)?.[1];
            const data = item.match(/^data: (.+)$/m)?.[1];
            if (!event || !data) continue;
            const parsed = JSON.parse(data) as Record<string, unknown>;
            if (event === "turn_completed") return parsed as AssistantMessageResult;
            if (event === "turn_failed") throw new Error(String(parsed.detail ?? "Assistant turn failed."));
            onEvent(event as AssistantMessageStreamEvent, parsed);
        }
        if (done) break;
    }
    throw new Error("Assistant stream ended before the turn completed.");
}

export async function postAssistantSynthesize(
    corpusId: string,
    payload: {
        query: string;
        document_ids?: string[] | null;
        async_mode?: boolean | null;
    }
): Promise<AssistantSynthesizeResult> {
    return apiFetch(`${BASE}/corpora/${corpusId}/assistant/synthesize`, {
        method: "POST",
        body: JSON.stringify(payload),
    });
}

export async function listResearchMemos(
    projectId: string,
    corpusId?: string,
    includeArchived = false
): Promise<ResearchMemo[]> {
    const params = new URLSearchParams();
    if (corpusId) params.set("corpus_id", corpusId);
    if (includeArchived) params.set("include_archived", "true");
    const query = params.size ? `?${params}` : "";
    return apiFetch(`${BASE}/projects/${projectId}/memos${query}`);
}

export async function getResearchMemo(memoId: string): Promise<ResearchMemo> {
    return apiFetch(`${BASE}/memos/${memoId}`);
}

export async function createResearchMemo(
    projectId: string,
    payload: {
        corpus_id?: string | null;
        title: string;
        body: string;
        source_type?: string;
        citations?: Array<Record<string, unknown>>;
        claims?: Array<Record<string, unknown>>;
        provenance?: Record<string, unknown>;
        evidence_revision_hash?: string | null;
        originating_assistant_message_id?: string | null;
        originating_synthesis_run_id?: string | null;
    }
): Promise<ResearchMemo> {
    return apiFetch(`${BASE}/projects/${projectId}/memos`, {
        method: "POST",
        body: JSON.stringify(payload),
    });
}

export async function saveAssistantAsResearchMemo(
    corpusId: string,
    payload: { message_id: string; title: string; body: string }
): Promise<ResearchMemo> {
    return apiFetch(`${BASE}/corpora/${corpusId}/memos/from-assistant`, {
        method: "POST",
        body: JSON.stringify(payload),
    });
}

export async function saveSynthesisAsResearchMemo(
    corpusId: string,
    payload: { run_id: string; title: string; body: string }
): Promise<ResearchMemo> {
    return apiFetch(`${BASE}/corpora/${corpusId}/memos/from-synthesis`, {
        method: "POST",
        body: JSON.stringify(payload),
    });
}

export async function updateResearchMemo(
    memoId: string,
    payload: { title?: string; body?: string }
): Promise<ResearchMemo> {
    return apiFetch(`${BASE}/memos/${memoId}`, {
        method: "PATCH",
        body: JSON.stringify(payload),
    });
}

export async function archiveResearchMemo(memoId: string): Promise<ResearchMemo> {
    return apiFetch(`${BASE}/memos/${memoId}/archive`, { method: "POST" });
}

export async function assistantRetrieve(
    corpusId: string,
    payload: {
        query: string;
        intent?: string | null;
        document_ids?: string[] | null;
        top_k?: number | null;
        retrieval_mode?: "dense" | "semantic" | "lexical" | "hybrid" | null;
    }
): Promise<{
    scope: AssistantScope;
    retrieval_trace_id: string | null;
    chunks: Array<{
        chunk_id: string;
        document_id: string;
        content: string;
        score: number;
        filename: string;
        chunk_index: number;
        page_number: number | null;
        rank: number | null;
        retrieval_sources: string[];
    }>;
    degraded: boolean;
    degradation_reason: string | null;
    no_matches: boolean;
    coverage: AssistantCoverage;
    intent: string | null;
    fusion_method: string | null;
}> {
    return apiFetch(`${BASE}/corpora/${corpusId}/assistant/retrieve`, {
        method: "POST",
        body: JSON.stringify(payload),
    });
}
