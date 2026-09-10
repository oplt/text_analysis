import { apiFetch, type Paginated } from "./client";

const BASE = "/memory";

export type MemoryMetadata = {
    memory_level: string;
    memory_type: string;
    user_id: string;
    agent_id: string;
    run_id: string | null;
    project_id: string | null;
    source: string;
    source_ref: string | null;
    confidence: number;
    privacy: string;
    version: number;
    created_at: string | null;
    last_seen_at: string | null;
    last_confirmed_at: string | null;
};

export type MemoryItem = {
    id: string;
    content: string;
    metadata: MemoryMetadata;
    score: number | null;
    created_at: string | null;
    updated_at: string | null;
};

export type MemoryAuditLog = {
    id: string;
    user_id: string | null;
    agent_id: string | null;
    run_id: string | null;
    project_id: string | null;
    operation: string;
    external_memory_id: string | null;
    memory_level: string | null;
    memory_type: string | null;
    source_ref: string | null;
    created_at: string;
};

export type MemoryWriteResult = {
    memory_id: string;
    memory_level: string;
    memory_type: string;
    accepted: boolean;
    rejection_reason: string | null;
};

export async function listMemories(options: {
    memoryLevel?: string;
    projectId?: string;
    agentId?: string;
    limit?: number;
    offset?: number;
} = {}): Promise<Paginated<MemoryItem>> {
    const search = new URLSearchParams();
    if (options.memoryLevel) search.set("memory_level", options.memoryLevel);
    if (options.projectId) search.set("project_id", options.projectId);
    if (options.agentId) search.set("agent_id", options.agentId);
    search.set("limit", String(options.limit ?? 20));
    search.set("offset", String(options.offset ?? 0));
    return apiFetch(`${BASE}?${search}`);
}

export async function searchMemories(payload: {
    query: string;
    agent_id?: string;
    run_id?: string;
    project_id?: string;
    memory_levels?: string[];
    limit?: number;
}): Promise<MemoryItem[]> {
    return apiFetch(`${BASE}/search`, {
        method: "POST",
        body: JSON.stringify({
            agent_id: "default",
            limit: 10,
            ...payload,
        }),
    });
}

export async function getMemory(memoryId: string): Promise<MemoryItem> {
    return apiFetch(`${BASE}/${encodeURIComponent(memoryId)}`);
}

export async function deleteMemory(memoryId: string): Promise<void> {
    await apiFetch(`${BASE}/${encodeURIComponent(memoryId)}`, { method: "DELETE" });
}

export async function forgetMemory(memoryId: string, reason: string): Promise<void> {
    await apiFetch(`${BASE}/forget/${encodeURIComponent(memoryId)}`, {
        method: "POST",
        body: JSON.stringify({ reason }),
    });
}

export async function listMemoryAuditLogs(options: {
    limit?: number;
    offset?: number;
} = {}): Promise<Paginated<MemoryAuditLog>> {
    const search = new URLSearchParams();
    search.set("limit", String(options.limit ?? 20));
    search.set("offset", String(options.offset ?? 0));
    return apiFetch(`${BASE}/audit?${search}`);
}

export async function createMemory(payload: {
    content: string;
    memory_level: string;
    memory_type?: string;
    agent_id?: string;
    project_id?: string;
    source?: string;
    source_ref?: string;
    confidence?: number;
    privacy?: string;
}): Promise<MemoryWriteResult> {
    return apiFetch(`${BASE}`, {
        method: "POST",
        body: JSON.stringify({
            agent_id: "default",
            source: "manual",
            confidence: 0.85,
            privacy: "normal",
            ...payload,
        }),
    });
}
