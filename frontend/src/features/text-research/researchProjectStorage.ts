const LAST_RESEARCH_PROJECT_KEY = "text-research:last-project-id";

export function getLastResearchProjectId(): string | null {
    try {
        return localStorage.getItem(LAST_RESEARCH_PROJECT_KEY);
    } catch {
        return null;
    }
}

export function setLastResearchProjectId(projectId: string): void {
    if (!projectId) return;
    try {
        localStorage.setItem(LAST_RESEARCH_PROJECT_KEY, projectId);
    } catch {
        // Ignore quota / private-mode failures.
    }
}

export function researchLabPath(projectId: string, tab = "dashboard"): string {
    return `/research/${projectId}/${tab}`;
}
