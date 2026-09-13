import { isActiveCanonicalStatus } from "../../components/ui/runStatusModel";

export function isActiveRunStatus(status?: string | null): boolean {
    return isActiveCanonicalStatus(status);
}

/** Poll only active jobs while a live SSE connection is unavailable. */
export function activeRunRefetchInterval(query: {
    state: { data?: { status?: string } | null };
}, sseConnected = false): number | false {
    return !sseConnected && isActiveRunStatus(query.state.data?.status) ? 2000 : false;
}
