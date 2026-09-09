export function collectScientificWarnings(
    ...sources: Array<Record<string, unknown> | null | undefined>
): string[] {
    const out: string[] = [];
    for (const source of sources) {
        if (!source) continue;
        for (const key of ["scientific_warnings", "diagnostics", "warnings", "evaluation_messages"]) {
            const value = source[key];
            if (!Array.isArray(value)) continue;
            for (const item of value) {
                const text = String(item ?? "").trim();
                if (text && !out.includes(text)) out.push(text);
            }
        }
    }
    return out;
}
