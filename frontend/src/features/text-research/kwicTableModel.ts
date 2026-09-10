export type KwicSearchRow = {
    left: string;
    keyword: string;
    right: string;
    document: string;
    organization: string;
    /** Lowercased concatenation for filter matching — avoid JSON.stringify per keystroke */
    searchText: string;
};

function asRecord(value: unknown): Record<string, unknown> | null {
    if (value && typeof value === "object" && !Array.isArray(value)) {
        return value as Record<string, unknown>;
    }
    return null;
}

function pickString(row: Record<string, unknown>, keys: string[]): string {
    for (const key of keys) {
        const value = row[key];
        if (typeof value === "string") return value;
        if (typeof value === "number" || typeof value === "boolean") return String(value);
    }
    return "";
}

/** Build a memoizable searchable representation of KWIC matches. */
export function toKwicSearchRows(matches: unknown[]): KwicSearchRow[] {
    return matches.map((entry) => {
        const row = asRecord(entry) ?? {};
        const left = pickString(row, ["left_context", "left"]);
        const keyword = pickString(row, ["keyword", "match", "term"]);
        const right = pickString(row, ["right_context", "right"]);
        const document = pickString(row, ["document_title", "title", "document"]);
        const organization = pickString(row, ["organization"]);
        const searchText = [left, keyword, right, document, organization]
            .join(" ")
            .toLocaleLowerCase();
        return { left, keyword, right, document, organization, searchText };
    });
}

export function filterKwicRows(rows: KwicSearchRow[], filter: string): KwicSearchRow[] {
    const needle = filter.trim().toLocaleLowerCase();
    if (!needle) return rows;
    return rows.filter((row) => row.searchText.includes(needle));
}

export function kwicRowsToCsv(rows: KwicSearchRow[]): string {
    const header = "left,keyword,right,document,organization";
    const body = rows.map((row) =>
        [row.left, row.keyword, row.right, row.document, row.organization]
            .map((cell) => JSON.stringify(cell ?? ""))
            .join(",")
    );
    return [header, ...body].join("\n");
}
