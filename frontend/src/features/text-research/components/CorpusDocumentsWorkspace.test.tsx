import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ThemeProvider, createTheme } from "@mui/material/styles";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { CorpusDocumentsWorkspace } from "./CorpusDocumentsWorkspace";
import { DEFAULT_DOCUMENT_FILTERS } from "./corpusDocumentFiltersModel";
import type { CorpusDocument } from "../types";

function doc(id: string, title: string): CorpusDocument {
    return {
        id,
        corpus_id: "c1",
        rag_document_id: null,
        title,
        organization: "Org",
        organization_type: null,
        publication_year: 2021,
        publication_type: null,
        country: "DE",
        region: null,
        cultural_sphere: null,
        language: "de",
        education_level: null,
        source_url: null,
        research_notes: null,
        metadata_json: null,
        created_at: "2026-01-01T00:00:00Z",
        updated_at: "2026-01-01T00:00:00Z",
    };
}

function renderWorkspace(overrides: Partial<Parameters<typeof CorpusDocumentsWorkspace>[0]> = {}) {
    const client = new QueryClient({
        defaultOptions: { queries: { retry: false } },
    });
    const onSelectDoc = vi.fn();
    const props = {
        corpusId: "c1",
        filters: DEFAULT_DOCUMENT_FILTERS,
        onFiltersChange: vi.fn(),
        page: 0,
        onPageChange: vi.fn(),
        pageSize: 25,
        documents: [doc("1", "Alpha"), doc("2", "Beta")],
        total: 2,
        isLoading: false,
        isError: false,
        error: null,
        onRetry: vi.fn(),
        selectedIds: [] as string[],
        onToggleSelected: vi.fn(),
        onToggleAllOnPage: vi.fn(),
        selectedDoc: doc("1", "Alpha"),
        onSelectDoc,
        onOpenCreateCorpus: vi.fn(),
        onBulkEdit: vi.fn(),
        unitType: "paragraph" as const,
        unitCountForType: 10,
        ...overrides,
    };
    render(
        <QueryClientProvider client={client}>
            <ThemeProvider theme={createTheme()}>
                <CorpusDocumentsWorkspace {...props} />
            </ThemeProvider>
        </QueryClientProvider>
    );
    return { onSelectDoc };
}

describe("CorpusDocumentsWorkspace", () => {
    it("renders the document master list with sticky table chrome", () => {
        renderWorkspace();
        expect(screen.getByRole("grid", { name: "Corpus documents" })).toBeInTheDocument();
        expect(screen.getAllByText("Alpha").length).toBeGreaterThan(0);
        expect(screen.getByText("Beta")).toBeInTheDocument();
        expect(screen.getByLabelText("Table density")).toBeInTheDocument();
    });

    it("moves selection with arrow keys", async () => {
        const user = userEvent.setup();
        const { onSelectDoc } = renderWorkspace();
        const grid = screen.getByRole("grid", { name: "Corpus documents" });
        grid.focus();
        await user.keyboard("{ArrowDown}");
        expect(onSelectDoc).toHaveBeenCalledWith(expect.objectContaining({ id: "2", title: "Beta" }));
    });
});
