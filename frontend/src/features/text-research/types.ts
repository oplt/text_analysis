export type ResearchCorpus = {
    id: string;
    project_id: string;
    name: string;
    description: string | null;
    created_by: string;
    created_at: string;
    updated_at: string;
};

export type CorpusDocument = {
    id: string;
    corpus_id: string;
    rag_document_id: string;
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
    created_by: string;
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

export type Annotation = {
    id: string;
    text_unit_id: string;
    label_id: string;
    annotator_id: string;
    value: string;
    confidence: number | null;
    comment: string | null;
    codebook_version: string;
    created_at: string;
    updated_at: string;
};

export type AnnotationQueueItem = {
    task: {
        id: string;
        text_unit_id: string;
        annotator_id: string;
        status: string;
        created_at: string;
        updated_at: string;
    };
    text_unit: {
        id: string;
        corpus_id: string;
        corpus_document_id: string;
        unit_type: string;
        unit_index: number;
        text: string;
        token_count: number | null;
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

export type DatasetPreview = {
    unit_count: number;
    document_count: number;
    unit_ids: string[];
    document_ids: string[];
    class_distribution: Record<string, Record<string, number>>;
    unit_labels: Record<string, string[]>;
    missing_labels: number;
    excluded_disagreements: number;
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
    { slug: "reliability", label: "Reliability" },
    { slug: "analysis", label: "Analysis" },
    { slug: "classification", label: "Classification" },
    { slug: "topics", label: "Topics" },
    { slug: "robustness", label: "Robustness" },
    { slug: "explorer", label: "Explorer" },
    { slug: "runs", label: "Runs & Exports" },
] as const;

export type ResearchTabSlug = (typeof RESEARCH_TABS)[number]["slug"];
