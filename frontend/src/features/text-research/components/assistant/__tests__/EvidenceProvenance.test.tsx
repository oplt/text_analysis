import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { EvidenceProvenance } from "../EvidenceProvenance";

describe("EvidenceProvenance", () => {
    it("distinguishes a fixed degraded snapshot from no results", () => {
        render(
            <EvidenceProvenance
                scope={{
                    corpus_id: "corpus",
                    project_id: "project",
                    corpus_name: "Corpus",
                    rag_document_ids: [],
                    corpus_document_ids: [],
                    indexed_rag_document_ids: [],
                    unavailable_rag_document_ids: [],
                    scope_hash: "scope",
                    evidence_revision_hash: "abcdef1234567890",
                    index_version: null,
                    retrieval_version: null,
                    total_documents: 2,
                    indexed_count: 1,
                    unavailable_count: 1,
                    unavailable_corpus_document_ids: ["missing"],
                    unavailable_reasons: {},
                    documents: [],
                    document_bindings: [],
                    warnings: [],
                    scope_mode: "fixed",
                }}
                coverage={{
                    documents_in_scope: 2,
                    documents_with_retrieved_evidence: 0,
                    retrieved_passage_count: 0,
                    coverage_ratio: 0,
                }}
                evidenceRevisionHash="abcdef1234567890"
                degraded
                degradationReason="dense_branch_failed"
                noResults
                citations={[]}
            />
        );

        expect(screen.getByLabelText("Evidence provenance")).toHaveTextContent("Fixed snapshot");
        expect(screen.getByText(/No relevant evidence/)).toBeInTheDocument();
        expect(screen.getByText(/Retrieval was degraded/)).toBeInTheDocument();
    });
});
