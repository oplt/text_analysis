import { describe, expect, it } from "vitest";
import { toMetadataDraft } from "./CorpusDocumentInspector";
import type { CorpusDocument } from "../types";

function doc(overrides: Partial<CorpusDocument> = {}): CorpusDocument {
    return {
        id: "d1",
        corpus_id: "c1",
        rag_document_id: "rag-1",
        title: "Treaty notes",
        organization: "UNESCO",
        organization_type: null,
        publication_year: 2020,
        publication_type: null,
        country: "FR",
        region: null,
        cultural_sphere: null,
        language: "en",
        education_level: null,
        source_url: "https://example.org",
        research_notes: "Primary source",
        metadata_json: null,
        created_at: "2026-01-01T00:00:00Z",
        updated_at: "2026-01-02T00:00:00Z",
        ...overrides,
    };
}

describe("CorpusDocumentInspector helpers", () => {
    it("maps document fields into an editable draft", () => {
        expect(toMetadataDraft(doc())).toMatchObject({
            title: "Treaty notes",
            organization: "UNESCO",
            publication_year: "2020",
            language: "en",
            source_url: "https://example.org",
            research_notes: "Primary source",
        });
    });

    it("uses empty strings for missing optional fields", () => {
        expect(
            toMetadataDraft(
                doc({
                    title: null,
                    organization: null,
                    publication_year: null,
                    source_url: null,
                    research_notes: null,
                })
            )
        ).toMatchObject({
            title: "",
            organization: "",
            publication_year: "",
            source_url: "",
            research_notes: "",
        });
    });
});
