export function isActiveRunStatus(status?: string | null): boolean {
    return status === "queued" || status === "running" || status === "pending";
}

/** Poll only active jobs; 2s while active, false otherwise. */
export function activeRunRefetchInterval(query: {
    state: { data?: { status?: string } | null };
}): number | false {
    return isActiveRunStatus(query.state.data?.status) ? 2000 : false;
}
