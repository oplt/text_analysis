/** Resolve corpus/codebook selection against live server IDs and localStorage. */

export function resolveResearchSelectionId(options: {
    projectId: string;
    storageKey: (projectId: string) => string;
    availableIds: string[];
    currentId: string;
}): string {
    const { projectId, storageKey, availableIds, currentId } = options;
    const key = storageKey(projectId);

    if (availableIds.length === 0) {
        localStorage.removeItem(key);
        return "";
    }

    const stored = localStorage.getItem(key);
    if (stored && availableIds.includes(stored)) {
        return stored;
    }

    if (stored) {
        localStorage.removeItem(key);
    }

    if (currentId && availableIds.includes(currentId)) {
        localStorage.setItem(key, currentId);
        return currentId;
    }

    const fallback = availableIds[0] ?? "";
    if (fallback) {
        localStorage.setItem(key, fallback);
    } else {
        localStorage.removeItem(key);
    }
    return fallback;
}
