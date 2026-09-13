/**
 * Client-side RAG evaluation summaries (Phase 14).
 * Metrics are descriptive of a single ask/retrieve response — not offline benchmarks.
 */

export type RagCitationLike = {
    document_id: string;
    chunk_id: string;
    score: number;
};

export type RagAskLike = {
    citations: RagCitationLike[];
    retrieved_chunk_ids: string[];
    no_context_found: boolean;
    retrieval_degraded: boolean;
    memory_degraded: boolean;
    injection_chunks_filtered: number;
};

export type RagEvaluationSummary = {
    retrievedCount: number;
    citationCount: number;
    /** Share of retrieved chunks that appear as citations (0–1). */
    citationCoverage: number | null;
    distinctDocuments: number;
    meanCitationScore: number | null;
    groundingNotes: string[];
};

export function summarizeRagAsk(result: RagAskLike): RagEvaluationSummary {
    const retrievedCount = result.retrieved_chunk_ids.length;
    const citationCount = result.citations.length;
    const citedIds = new Set(result.citations.map((c) => c.chunk_id));
    const covered =
        retrievedCount > 0
            ? [...citedIds].filter((id) => result.retrieved_chunk_ids.includes(id)).length /
              retrievedCount
            : null;
    const scores = result.citations.map((c) => c.score).filter((s) => Number.isFinite(s));
    const meanCitationScore =
        scores.length > 0 ? scores.reduce((a, b) => a + b, 0) / scores.length : null;
    const distinctDocuments = new Set(result.citations.map((c) => c.document_id)).size;

    const groundingNotes: string[] = [];
    if (result.no_context_found) {
        groundingNotes.push("No retrieval context was available for this answer.");
    }
    if (result.retrieval_degraded) {
        groundingNotes.push("Retrieval ran in a degraded mode — treat citations cautiously.");
    }
    if (result.memory_degraded) {
        groundingNotes.push("Memory lookup degraded; answer may omit memory evidence.");
    }
    if (result.injection_chunks_filtered > 0) {
        groundingNotes.push(
            `Filtered ${result.injection_chunks_filtered} injection chunk(s) before answering.`
        );
    }
    if (citationCount === 0 && !result.no_context_found) {
        groundingNotes.push("Answer returned without citations — grounding is unverified.");
    }
    if (covered != null && covered < 0.5 && citationCount > 0) {
        groundingNotes.push(
            "Fewer than half of retrieved chunks were cited — review unused evidence."
        );
    }

    return {
        retrievedCount,
        citationCount,
        citationCoverage: covered,
        distinctDocuments,
        meanCitationScore,
        groundingNotes,
    };
}

export function formatCoverage(value: number | null): string {
    if (value == null || !Number.isFinite(value)) return "—";
    return `${(value * 100).toFixed(0)}%`;
}

/** Pull indexing/provenance fields from chunk metadata when present. */
export function indexingParamsFromChunkMeta(
    meta: Record<string, unknown> | null | undefined
): Array<{ key: string; label: string; value: string; helpTermId?: string }> {
    if (!meta) return [];
    const rows: Array<{ key: string; label: string; value: string; helpTermId?: string }> = [];
    const push = (key: string, label: string, helpTermId?: string) => {
        const raw = meta[key];
        if (raw == null || raw === "") return;
        rows.push({ key, label, value: String(raw), helpTermId });
    };
    push("embedding_model", "Embedding model", "embeddings");
    push("embedding_provider", "Embedding provider", "embeddings");
    push("embedding_dimensions", "Embedding dimensions", "embeddings");
    push("chunk_size", "Chunk size", "chunk_size");
    push("chunk_overlap", "Chunk overlap", "chunk_overlap");
    push("chunker_version", "Chunker version");
    push("parser_version", "Parser version");
    push("chunk_policy_version", "Chunk policy");
    return rows;
}

export function documentPipelineLabel(status: string): {
    parsing: string;
    indexing: string;
} {
    switch (status) {
        case "uploaded":
            return { parsing: "Queued", indexing: "Not started" };
        case "processing":
            return { parsing: "In progress", indexing: "In progress" };
        case "indexed":
            return { parsing: "Complete", indexing: "Indexed" };
        case "failed":
            return { parsing: "Failed / unknown", indexing: "Failed" };
        default:
            return { parsing: status, indexing: status };
    }
}
