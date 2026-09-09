export function isActiveRunStatus(status?: string | null): boolean {
    return status === "queued" || status === "running" || status === "pending";
}

/** Poll only active jobs while a live SSE connection is unavailable. */
export function activeRunRefetchInterval(query: {
    state: { data?: { status?: string } | null };
}, sseConnected = false): number | false {
    return !sseConnected && isActiveRunStatus(query.state.data?.status) ? 2000 : false;
}
