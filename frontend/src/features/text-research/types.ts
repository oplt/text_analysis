export type ResearchCorpus = {
    id: string;
    project_id: string;
    name: string;
    description: string | null;
    created_by: string;
    created_at: string;
    updated_at: string;
};

export type ResearchMemo = {
    id: string;
    project_id: string;
    corpus_id: string | null;
    user_id: string;
    source_type: "manual" | "assistant_answer" | "analysis_result" | string;
    status: "active" | "archived" | string;
    title: string;
    body: string;
    originating_assistant_message_id: string | null;
    originating_synthesis_run_id: string | null;
    evidence_revision_hash: string | null;
    citations: Array<Record<string, unknown>>;
    claims: Array<Record<string, unknown>>;
    provenance: Record<string, unknown>;
    created_at: string;
    updated_at: string;
    archived_at: string | null;
};

export type CorpusDocument = {
    id: string;
    corpus_id: string;
    rag_document_id: string | null;
    title: string | null;
    organization: string | null;
    organization_type: string | null;
    publication_year: number | null;
    publication_type: string | null;
    country: string | null;
    region: string | null;
    cultural_sphere: string | null;
    language: string | null;
    education_level: string | null;
    source_url: string | null;
    research_notes: string | null;
    metadata_json: Record<string, unknown> | null;
    created_at: string;
    updated_at: string;
};

export type Codebook = {
    id: string;
    project_id: string;
    name: string;
    description: string | null;
    version: string;
    is_frozen: boolean;
    created_by: string;
    created_at: string;
};

export type AnnotationLabel = {
    id: string;
    codebook_id: string;
    name: string;
    description: string | null;
    inclusion_criteria: string | null;
    exclusion_criteria: string | null;
    positive_examples: string[] | null;
    negative_examples: string[] | null;
    is_placeholder: boolean;
    created_at: string;
};

export type AnalysisRun = {
    id: string;
    project_id: string;
    corpus_id: string | null;
    run_type: string;
    status: string;
    run_version?: number;
    evidence_revision_hash?: string | null;
    progress_stage: string | null;
    parameters: Record<string, unknown> | null;
    metrics: Record<string, unknown> | null;
    results: Record<string, unknown> | null;
    artifact_path: string | null;
    random_seed: number | null;
    created_by: string;
    started_at: string | null;
    completed_at: string | null;
    error_message: string | null;
    created_at: string;
    /** Whether one-click Reproduce can exactly re-execute this run. */
    rerunnable?: boolean;
    /** Human-readable reason when ``rerunnable`` is false. */
    rerun_block_reason?: string | null;
};

export type TrainedModel = {
    id: string;
    project_id: string;
    corpus_id: string;
    analysis_run_id: string;
    training_dataset_snapshot_id: string;
    model_family: string;
    task_type: string;
    label_ids: string[];
    feature_config: Record<string, unknown>;
    training_config: Record<string, unknown>;
    metrics: Record<string, unknown>;
    version: number;
    name: string | null;
    lifecycle_status: "candidate" | "staging" | "production" | "deprecated" | "archived" | string;
    lifecycle_notes: string | null;
    lifecycle_updated_at: string | null;
    created_by: string;
    created_at: string;
};

export type ModelLifecycleStatus = "candidate" | "staging" | "production" | "deprecated" | "archived";

export type PredictionSetSummary = {
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
};

export type ModelPredictionItem = {
    id: string;
    trained_model_id: string;
    text_unit_id: string;
    predicted_labels: string[];
    scores: Record<string, number>;
    uncertainty: number | null;
    created_at: string;
};

export type UncertainPrediction = {
    prediction: {
        id: string;
        trained_model_id: string;
        text_unit_id: string;
        predicted_labels: string[];
        scores: Record<string, number>;
        uncertainty: number | null;
        created_at: string;
    };
    text_unit: {
        id: string;
        corpus_document_id: string;
        unit_type: string;
        position: number;
        text: string;
    };
};

export type TrainingDatasetSnapshot = {
    id: string;
    project_id: string;
    corpus_id: string;
    name: string;
    unit_type: string;
    codebook_id: string;
    codebook_version: string;
    annotation_source: string;
    minimum_agreement: number | null;
    annotation_campaign_id?: string | null;
    annotation_campaign_snapshot_hash?: string | null;
    adjudication_policy?: string | null;
    gold_source?: string | null;
    created_by: string;
    created_at: string;
};

export type PreprocessingProfile = {
    id: string;
    project_id: string;
    name: string;
    description: string | null;
    config: Record<string, unknown>;
    created_by: string;
    created_at: string;
    updated_at: string;
};

export type CleaningProfile = {
    id: string;
    project_id: string;
    name: string;
    description: string | null;
    version: string;
    config: Record<string, unknown>;
    created_by: string;
    created_at: string;
    updated_at: string;
};

export type CleaningPreview = {
    rows: Array<{
        raw_preview: string;
        cleaned_preview: string;
        chars_raw: number;
        chars_cleaned: number;
        raw_checksum: string;
        cleaned_checksum: string;
        steps: Array<{
            name: string;
            discarded_chars: number;
            details: Record<string, unknown>;
        }>;
    }>;
    config: Record<string, unknown>;
    engine: string;
    engine_version: string;
    profile_name: string | null;
    profile_version: string | null;
};

export type IngestionQaDocument = {
    document_id: string;
    title: string | null;
    language: string | null;
    findings: Array<{
        code: string;
        severity: "error" | "warning" | "info" | string;
        message: string;
        details: Record<string, unknown>;
    }>;
    metrics: Record<string, unknown>;
};

export type Annotation = {
    id: string;
    text_unit_id: string;
    label_id: string;
    annotator_id: string;
    value: string;
    confidence: number | null;
    comment: string | null;
    codebook_version: string;
    campaign_id?: string | null;
    created_at: string;
    updated_at: string;
};

export type AnnotationMode = "blind_reliability" | "ai_assisted";

export type AnnotationCampaign = {
    id: string;
    project_id: string;
    corpus_id: string;
    name: string;
    description: string | null;
    codebook_id: string | null;
    codebook_version: string | null;
    unit_type: string;
    sampling_strategy: string;
    assignment_strategy: string;
    sample_size: number | null;
    overlap_count: number | null;
    overlap_percent: number | null;
    blind_mode: boolean;
    ai_assistance_enabled: boolean;
    annotation_mode: AnnotationMode;
    reveal_after: string;
    status: string;
    annotator_ids: string[];
    created_by: string;
    created_at: string;
    started_at: string | null;
    completed_at: string | null;
    metadata: Record<string, unknown>;
};

export type AnnotationBlindPolicy = {
    blind_mode: boolean;
    hide_model_predictions: boolean;
    hide_peer_annotations?: boolean;
    hide_adjudications?: boolean;
    reveal_after?: string;
    released?: boolean;
    ai_assistance_enabled: boolean;
};

export type AnnotationQueueCampaignSummary = {
    id: string;
    name: string;
    annotation_mode: AnnotationMode;
    blind_mode: boolean;
    ai_assistance_enabled: boolean;
};

export type AnnotationQueueItem = {
    task: {
        id: string;
        text_unit_id: string;
        campaign_id?: string | null;
        annotator_id: string;
        status: string;
        assigned_at: string;
        completed_at: string | null;
    };
    campaign?: AnnotationQueueCampaignSummary | null;
    blind_policy?: AnnotationBlindPolicy | null;
    text_unit: {
        id: string;
        corpus_document_id: string;
        unit_type: string;
        position: number;
        page_number: number | null;
        paragraph_number: number | null;
        sentence_number: number | null;
        text: string;
        text_hash: string;
        created_at: string;
    } | null;
};

export type DashboardSummary = {
    corpus: { id: string; name: string };
    document_count: number;
    text_unit_counts: Record<string, number>;
    codebook_count: number;
    training_dataset_snapshot_count: number;
    trained_model_count: number;
    latest_model: {
        id: string;
        name: string | null;
        version: number;
        metrics: Record<string, unknown>;
    } | null;
    analysis_run_counts_by_type: Record<string, number>;
    analysis_run_counts_by_status: Record<string, number>;
    latest_reliability: {
        run_id: string;
        metrics: Record<string, unknown>;
    } | null;
    annotation_task_count: number;
    annotation_completed_count: number;
    annotation_completion_rate: number;
};

export type AnnotationProgress = {
    total_units: number;
    total_tasks: number;
    completed_tasks: number;
    completion_rate: number;
    units_with_completed_annotation: number;
    by_annotator: Record<string, { assigned: number; completed: number }>;
};

export type DatasetLabelGap = {
    text_unit_id: string;
    label_id: string;
};

export type DatasetPreview = {
    unit_count: number;
    document_count: number;
    unit_ids: string[];
    document_ids: string[];
    class_distribution: Record<string, Record<string, number>>;
    unit_labels: Record<string, string[]>;
    missing_labels: DatasetLabelGap[];
    excluded_disagreements: DatasetLabelGap[];
    annotator_coverage: string[];
    warnings: string[];
};

export type ExportManifest = {
    manifest: Record<string, unknown>;
};

export type UnitType = "document" | "paragraph" | "sentence";

export const UNIT_TYPE_OPTIONS: { value: UnitType; label: string }[] = [
    { value: "document", label: "Document" },
    { value: "paragraph", label: "Paragraph" },
    { value: "sentence", label: "Sentence" },
];

export const RESEARCH_TABS = [
    { slug: "dashboard", label: "Dashboard" },
    { slug: "corpus", label: "Corpus" },
    { slug: "annotation", label: "Annotation" },
    { slug: "codebook", label: "Codebook" },
    { slug: "reliability", label: "Reliability" },
    { slug: "analysis", label: "Analysis" },
    { slug: "dictionaries", label: "Dictionaries" },
    { slug: "comparative", label: "Comparative" },
    { slug: "classification", label: "Classification" },
    { slug: "active-learning", label: "Active Learning" },
    { slug: "models", label: "Model Registry" },
    { slug: "predictions", label: "Predictions" },
    { slug: "drift", label: "Drift" },
    { slug: "topics", label: "Topics" },
    { slug: "robustness", label: "Robustness" },
    { slug: "explorer", label: "Explorer" },
    { slug: "contextual", label: "Contextual" },
    { slug: "runs", label: "Runs" },
    { slug: "exports", label: "Exports" },
] as const;

export type ResearchTabSlug = (typeof RESEARCH_TABS)[number]["slug"];
