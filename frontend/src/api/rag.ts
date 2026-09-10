import { apiFetch, type Paginated } from "./client";

const BASE = "/rag";

export type RagDocument = {
    id: string;
    user_id: string;
    project_id: string | null;
    organization_id: string | null;
    filename: string;
    original_filename: string;
    content_type: string;
    storage_path: string | null;
    status: string;
    source_type: string;
    metadata: Record<string, unknown>;
    created_at: string;
    updated_at: string;
};

export type RagIngestionJob = {
    id: string;
    document_id: string;
    user_id: string;
    project_id: string | null;
    status: string;
    error_message: string | null;
    started_at: string | null;
    finished_at: string | null;
    created_at: string;
};

export type RagDocumentUpload = {
    document: RagDocument;
    ingestion_job: RagIngestionJob;
};

export type RagChunk = {
    id: string;
    document_id: string;
    chunk_index: number;
    content: string;
    token_count: number;
    metadata: Record<string, unknown>;
};

export type RagRetrievedChunk = {
    chunk_id: string;
    document_id: string;
    content: string;
    score: number;
    filename: string;
    chunk_index: number;
    page_number: number | null;
};

export type RagRetrieveResult = {
    chunks: RagRetrievedChunk[];
    degraded: boolean;
    degradation_reason: string | null;
    no_matches: boolean;
    injection_chunks_filtered: number;
};

export type RagCitation = {
    document_id: string;
    chunk_id: string;
    filename: string;
    score: number;
    snippet: string;
    page_number: number | null;
    chunk_index: number | null;
};

export type RagAskResult = {
    query: string;
    answer: string;
    citations: RagCitation[];
    retrieved_chunk_ids: string[];
    model_name: string;
    latency_ms: number;
    no_context_found: boolean;
    ai_run_id: string | null;
    retrieval_degraded: boolean;
    memory_degraded: boolean;
    degradation_reason: string | null;
    injection_chunks_filtered: number;
};

export type RagQueryHistoryItem = {
    id: string;
    query: string;
    answer: string;
    project_id: string | null;
    model_name: string;
    latency_ms: number;
    created_at: string;
};

export async function listRagDocuments(options: {
    projectId?: string;
    limit?: number;
    offset?: number;
} = {}): Promise<Paginated<RagDocument>> {
    const search = new URLSearchParams();
    if (options.projectId) search.set("project_id", options.projectId);
    search.set("limit", String(options.limit ?? 20));
    search.set("offset", String(options.offset ?? 0));
    return apiFetch(`${BASE}/documents?${search}`);
}

export async function getRagDocument(documentId: string): Promise<RagDocument> {
    return apiFetch(`${BASE}/documents/${encodeURIComponent(documentId)}`);
}

export async function uploadRagDocument(
    file: File,
    projectId?: string
): Promise<RagDocumentUpload> {
    const form = new FormData();
    form.append("file", file);
    if (projectId) form.append("project_id", projectId);
    return apiFetch(`${BASE}/documents/upload`, { method: "POST", body: form });
}

export async function deleteRagDocument(documentId: string): Promise<void> {
    await apiFetch(`${BASE}/documents/${encodeURIComponent(documentId)}`, {
        method: "DELETE",
    });
}

export async function indexRagDocument(documentId: string): Promise<RagIngestionJob> {
    return apiFetch(`${BASE}/documents/${encodeURIComponent(documentId)}/index`, {
        method: "POST",
    });
}

export async function listRagChunks(
    documentId: string,
    options: { limit?: number; offset?: number; contentMode?: "snippet" | "full" } = {}
): Promise<Paginated<RagChunk>> {
    const search = new URLSearchParams();
    search.set("limit", String(options.limit ?? 20));
    search.set("offset", String(options.offset ?? 0));
    search.set("content_mode", options.contentMode ?? "snippet");
    return apiFetch(
        `${BASE}/documents/${encodeURIComponent(documentId)}/chunks?${search}`
    );
}

export async function retrieveRagChunks(payload: {
    query: string;
    project_id?: string;
    document_ids?: string[];
    top_k?: number;
}): Promise<RagRetrieveResult> {
    return apiFetch(`${BASE}/retrieve`, {
        method: "POST",
        body: JSON.stringify(payload),
    });
}

export async function askRag(payload: {
    query: string;
    project_id?: string;
    run_id?: string;
    agent_id?: string;
    document_ids?: string[];
}): Promise<RagAskResult> {
    return apiFetch(`${BASE}/ask`, {
        method: "POST",
        body: JSON.stringify(payload),
    });
}

export async function listRagQueries(options: {
    limit?: number;
    offset?: number;
} = {}): Promise<Paginated<RagQueryHistoryItem>> {
    const search = new URLSearchParams();
    search.set("limit", String(options.limit ?? 20));
    search.set("offset", String(options.offset ?? 0));
    return apiFetch(`${BASE}/queries?${search}`);
}

export async function getRagIngestionJob(jobId: string): Promise<RagIngestionJob> {
    return apiFetch(`${BASE}/jobs/${encodeURIComponent(jobId)}`);
}
