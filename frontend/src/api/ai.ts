import { apiFetch, apiFetchItems } from "./client";

export type AiProvider = {
    key: string;
    label: string;
    supports_generation: boolean;
    supports_embeddings: boolean;
};

export type AiPromptTemplate = {
    id: string;
    key: string;
    name: string;
    description: string | null;
    is_active: boolean;
    active_version_id: string | null;
    created_at: string;
    updated_at: string;
};

export type AiVariableDefinition = {
    name: string;
    description: string | null;
    required: boolean;
};

export type AiPromptVersion = {
    id: string;
    prompt_template_id: string;
    version_number: number;
    provider_key: string;
    model_name: string;
    system_prompt: string;
    user_prompt_template: string;
    variable_definitions: AiVariableDefinition[];
    response_format: "text" | "json";
    temperature: number;
    rollout_percentage: number;
    is_published: boolean;
    input_cost_per_million: number;
    output_cost_per_million: number;
    created_by_user_id: string | null;
    created_at: string;
};

export type AiDocument = {
    id: string;
    title: string;
    description: string | null;
    filename: string | null;
    content_type: string;
    size_bytes: number;
    ingestion_status: string;
    metadata: Record<string, unknown>;
    chunk_count: number;
    created_at: string;
    updated_at: string;
};

export type AiRun = {
    id: string;
    prompt_template_id: string | null;
    prompt_version_id: string | null;
    provider_key: string;
    model_name: string;
    status: string;
    response_format: string;
    variables: Record<string, unknown>;
    retrieval_query: string | null;
    retrieved_chunk_ids: string[];
    retrieval_degraded: boolean;
    memory_degraded: boolean;
    degradation_reason: string | null;
    injection_chunks_filtered: number;
    input_messages: Array<{ role: string; content: string }>;
    output_text: string | null;
    output_json: Record<string, unknown> | null;
    latency_ms: number | null;
    input_tokens: number;
    output_tokens: number;
    total_tokens: number;
    estimated_cost_micros: number;
    error_message: string | null;
    review_status: string;
    created_at: string;
    completed_at: string | null;
};

export type AiReviewItem = {
    id: string;
    run_id: string;
    requested_by_user_id: string;
    assigned_to_user_id: string | null;
    reviewed_by_user_id: string | null;
    status: string;
    reviewer_notes: string | null;
    corrected_output: string | null;
    created_at: string;
    updated_at: string;
};

export type AiFeedback = {
    id: string;
    run_id: string;
    user_id: string;
    rating: number;
    comment: string | null;
    corrected_output: string | null;
    created_at: string;
};

export type AiEvaluationDataset = {
    id: string;
    name: string;
    description: string | null;
    created_at: string;
    updated_at: string;
};

export type AiEvaluationCase = {
    id: string;
    dataset_id: string;
    input_variables: Record<string, unknown>;
    retrieval_query: string | null;
    document_ids: string[];
    expected_chunk_ids: string[];
    expected_output_text: string | null;
    expected_output_json: Record<string, unknown> | null;
    notes: string | null;
    created_at: string;
};

export type AiEvaluationRunItem = {
    evaluation_case_id: string;
    ai_run_id: string;
    score: number;
    passed: boolean;
    notes: string | null;
};

export type AiEvaluationRun = {
    id: string;
    dataset_id: string;
    prompt_version_id: string;
    status: string;
    total_cases: number;
    passed_cases: number;
    average_score: number;
    created_at: string;
    completed_at: string | null;
    items: AiEvaluationRunItem[];
};

export type AiOverview = {
    providers: AiProvider[];
    prompt_templates: AiPromptTemplate[];
    recent_runs: AiRun[];
    documents: AiDocument[];
    datasets: AiEvaluationDataset[];
};

export type AiChunkMatch = {
    document_id: string;
    chunk_id: string;
    document_title: string;
    chunk_index: number;
    score: number;
    content: string;
};

export async function getAiOverview(): Promise<AiOverview> {
    return apiFetch("/ai/overview");
}

export async function listPromptVersions(templateId: string): Promise<AiPromptVersion[]> {
    return apiFetchItems<AiPromptVersion>(`/ai/prompts/${templateId}/versions`);
}

export async function createPromptTemplate(payload: {
    key: string;
    name: string;
    description?: string;
}): Promise<AiPromptTemplate> {
    return apiFetch("/ai/prompts", {
        method: "POST",
        body: JSON.stringify(payload),
    });
}

export async function updatePromptTemplate(
    templateId: string,
    payload: Partial<{
        name: string;
        description: string | null;
        is_active: boolean;
        active_version_id: string | null;
    }>
): Promise<AiPromptTemplate> {
    return apiFetch(`/ai/prompts/${templateId}`, {
        method: "PATCH",
        body: JSON.stringify(payload),
    });
}

export async function createPromptVersion(
    templateId: string,
    payload: {
        provider_key: string;
        model_name: string;
        system_prompt: string;
        user_prompt_template: string;
        variable_definitions: AiVariableDefinition[];
        response_format: "text" | "json";
        temperature: number;
        rollout_percentage: number;
        is_published: boolean;
        input_cost_per_million: number;
        output_cost_per_million: number;
    }
): Promise<AiPromptVersion> {
    return apiFetch(`/ai/prompts/${templateId}/versions`, {
        method: "POST",
        body: JSON.stringify(payload),
    });
}

export async function updatePromptVersion(
    templateId: string,
    versionId: string,
    payload: Partial<{
        is_published: boolean;
        rollout_percentage: number;
    }>
): Promise<AiPromptVersion> {
    return apiFetch(`/ai/prompts/${templateId}/versions/${versionId}`, {
        method: "PATCH",
        body: JSON.stringify(payload),
    });
}

export async function createAiDocument(payload: {
    title: string;
    description?: string;
    content: string;
    content_type?: string;
    metadata?: Record<string, unknown>;
}): Promise<AiDocument> {
    return apiFetch("/ai/documents", {
        method: "POST",
        body: JSON.stringify(payload),
    });
}

export async function uploadAiDocument(file: File, description?: string): Promise<AiDocument> {
    const formData = new FormData();
    formData.append("file", file);
    if (description) {
        formData.append("description", description);
    }
    return apiFetch("/ai/documents/upload", {
        method: "POST",
        body: formData,
    });
}

export async function retrieveAiChunks(payload: {
    query: string;
    document_ids: string[];
    top_k: number;
}): Promise<AiChunkMatch[]> {
    return apiFetch("/ai/retrieve", {
        method: "POST",
        body: JSON.stringify(payload),
    });
}

export async function createAiRun(payload: {
    prompt_template_key?: string;
    prompt_version_id?: string;
    variables: Record<string, unknown>;
    retrieval_query?: string;
    document_ids: string[];
    top_k: number;
    review_required: boolean;
}): Promise<AiRun> {
    return apiFetch("/ai/runs", {
        method: "POST",
        body: JSON.stringify(payload),
    });
}

export async function listAiReviews(): Promise<AiReviewItem[]> {
    return apiFetchItems<AiReviewItem>("/ai/reviews");
}

export async function createAiReview(
    runId: string,
    payload: { assigned_to_user_id?: string | null } = {}
): Promise<AiReviewItem> {
    return apiFetch(`/ai/runs/${runId}/reviews`, {
        method: "POST",
        body: JSON.stringify(payload),
    });
}

export async function decideAiReview(
    reviewId: string,
    payload: {
        status: "approved" | "rejected" | "changes_requested";
        reviewer_notes?: string;
        corrected_output?: string;
    }
): Promise<AiReviewItem> {
    return apiFetch(`/ai/reviews/${reviewId}/decision`, {
        method: "POST",
        body: JSON.stringify(payload),
    });
}

export async function listAiFeedback(runId: string): Promise<AiFeedback[]> {
    return apiFetchItems<AiFeedback>(`/ai/runs/${runId}/feedback`);
}

export async function createAiFeedback(
    runId: string,
    payload: { rating: -1 | 1; comment?: string; corrected_output?: string }
): Promise<AiFeedback> {
    return apiFetch(`/ai/runs/${runId}/feedback`, {
        method: "POST",
        body: JSON.stringify(payload),
    });
}

export async function listAiDatasets(): Promise<AiEvaluationDataset[]> {
    return apiFetchItems<AiEvaluationDataset>("/ai/evaluation-datasets");
}

export async function createAiDataset(payload: {
    name: string;
    description?: string;
}): Promise<AiEvaluationDataset> {
    return apiFetch("/ai/evaluation-datasets", {
        method: "POST",
        body: JSON.stringify(payload),
    });
}

export async function listAiDatasetCases(datasetId: string): Promise<AiEvaluationCase[]> {
    return apiFetchItems<AiEvaluationCase>(`/ai/evaluation-datasets/${datasetId}/cases`);
}

export async function createAiDatasetCase(
    datasetId: string,
    payload: {
        input_variables: Record<string, unknown>;
        retrieval_query?: string | null;
        document_ids?: string[];
        expected_chunk_ids?: string[];
        expected_output_text?: string | null;
        expected_output_json?: Record<string, unknown> | null;
        notes?: string | null;
    }
): Promise<AiEvaluationCase> {
    return apiFetch(`/ai/evaluation-datasets/${datasetId}/cases`, {
        method: "POST",
        body: JSON.stringify(payload),
    });
}

export async function listAiEvaluationRuns(): Promise<AiEvaluationRun[]> {
    return apiFetchItems<AiEvaluationRun>("/ai/evaluation-runs");
}

export async function runAiEvaluation(
    datasetId: string,
    promptVersionId: string
): Promise<AiEvaluationRun> {
    return apiFetch(`/ai/evaluation-datasets/${datasetId}/run`, {
        method: "POST",
        body: JSON.stringify({ prompt_version_id: promptVersionId }),
    });
}
