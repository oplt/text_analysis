const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000/api/v1";

let refreshPromise: Promise<boolean> | null = null;

function readCookie(name: string): string | null {
    const match = document.cookie.match(
        new RegExp(`(?:^|; )${name.replace(/[.*+?^${}()|[\\]\\\\]/g, "\\$&")}=([^;]*)`)
    );
    return match ? decodeURIComponent(match[1]) : null;
}

async function refreshAccessToken(): Promise<boolean> {
    try {
        const res = await fetch(`${API_BASE}/auth/refresh`, {
            method: "POST",
            credentials: "include",
            headers: buildCsrfHeaders(),
        });
        return res.ok;
    } catch {
        return false;
    }
}

function buildCsrfHeaders(): HeadersInit {
    const csrfToken = readCookie("csrf_token");
    return csrfToken ? { "X-CSRF-Token": csrfToken } : {};
}

export type Paginated<T> = {
    items: T[];
    total: number;
    limit: number;
    offset: number;
};

export async function apiFetch<T>(
    path: string,
    options: RequestInit = {},
    retry = true
): Promise<T> {
    const headers = new Headers(options.headers ?? {});
    const isFormData = options.body instanceof FormData;

    if (!isFormData && options.body !== undefined && !headers.has("Content-Type")) {
        headers.set("Content-Type", "application/json");
    }
    if (!headers.has("X-CSRF-Token")) {
        const csrfValue = readCookie("csrf_token");
        if (csrfValue) {
            headers.set("X-CSRF-Token", csrfValue);
        }
    }

    const response = await fetch(`${API_BASE}${path}`, {
        ...options,
        headers,
        credentials: "include",
    });

    if (response.status === 401 && retry) {
        // Deduplicate concurrent refresh attempts
        if (!refreshPromise) {
            refreshPromise = refreshAccessToken().finally(() => {
                refreshPromise = null;
            });
        }
        const refreshed = await refreshPromise;
        if (!refreshed) {
            throw new Error("Session expired. Please sign in again.");
        }
        return apiFetch<T>(path, options, false);
    }

    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: "Request failed" }));
        throw new Error(error.detail ?? "Request failed");
    }

    // Handle 204 No Content
    if (response.status === 204) return undefined as T;

    return response.json();
}

export async function apiFetchItems<T>(
    path: string,
    options: RequestInit = {}
): Promise<T[]> {
    const page = await apiFetch<Paginated<T>>(path, options);
    return page.items;
}
