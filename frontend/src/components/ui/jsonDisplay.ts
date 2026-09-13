/**
 * Helpers for Phase 18 — readable substitutes for raw JSON in primary UI.
 */

export function stringifyPretty(data: unknown): string {
    try {
        return JSON.stringify(data, null, 2);
    } catch {
        return String(data);
    }
}

export function formatDisplayValue(value: unknown): string {
    if (value == null) return "—";
    if (typeof value === "string") return value || "—";
    if (typeof value === "number" || typeof value === "boolean") return String(value);
    if (Array.isArray(value)) {
        if (!value.length) return "[]";
        if (value.every((item) => typeof item === "string" || typeof item === "number")) {
            return value.map(String).join(", ");
        }
        return `${value.length} items`;
    }
    if (typeof value === "object") {
        const keys = Object.keys(value as object);
        return keys.length ? `{${keys.slice(0, 4).join(", ")}${keys.length > 4 ? ", …" : ""}}` : "{}";
    }
    return String(value);
}

export type JsonKeyValueItem = {
    key: string;
    label: string;
    value: string;
};

/** Flatten one level of a plain object for KeyValueList / definition lists. */
export function recordToKeyValueItems(
    data: unknown,
    options?: { maxEntries?: number }
): JsonKeyValueItem[] {
    if (!data || typeof data !== "object" || Array.isArray(data)) return [];
    const max = options?.maxEntries ?? 24;
    return Object.entries(data as Record<string, unknown>)
        .slice(0, max)
        .map(([key, value]) => ({
            key,
            label: key.replaceAll("_", " "),
            value: formatDisplayValue(value),
        }));
}

export async function copyTextToClipboard(text: string): Promise<boolean> {
    try {
        if (navigator.clipboard?.writeText) {
            await navigator.clipboard.writeText(text);
            return true;
        }
    } catch {
        /* fall through */
    }
    try {
        const area = document.createElement("textarea");
        area.value = text;
        area.setAttribute("readonly", "");
        area.style.position = "absolute";
        area.style.left = "-9999px";
        document.body.appendChild(area);
        area.select();
        const ok = document.execCommand("copy");
        document.body.removeChild(area);
        return ok;
    } catch {
        return false;
    }
}
