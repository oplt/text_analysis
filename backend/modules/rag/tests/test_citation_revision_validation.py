from backend.modules.rag.application.citation_validation_service import CitationValidationService
from backend.modules.rag.domain.citation_validation_context import CitationValidationContext
from backend.modules.rag.domain.models import RetrievedChunk


def test_citation_rejects_chunk_outside_expected_index_revision():
    chunk = RetrievedChunk(
        chunk_id="chunk",
        document_id="document",
        content="evidence",
        score=1.0,
        filename="document.txt",
        chunk_index=0,
        index_revision_id="new-revision",
    )
    result = CitationValidationService().validate(
        raw_output='{"claims":[{"text":"claim","chunk_ids":["chunk"]}],"no_evidence":false}',
        retrieved_chunks=[chunk],
        allowed_document_ids=["document"],
        expected_index_revision_ids=["old-revision"],
    )
    assert result.citation_validation_failed
    assert result.claims == []
    assert result.citation_validation_status == "revision_mismatch"


def test_citation_context_binds_revision_by_document():
    chunk = RetrievedChunk(
        chunk_id="chunk",
        document_id="document",
        content="evidence",
        score=1.0,
        filename="document.txt",
        chunk_index=0,
        index_revision_id="rev-b",
    )
    context = CitationValidationContext(
        user_id="user",
        project_id="project",
        evidence_revision_hash=None,
        index_revision_ids=["rev-a", "rev-b"],
        revision_by_document={"document": "rev-a"},
        allowed_document_ids=["document"],
    )
    result = CitationValidationService().validate(
        raw_output='{"claims":[{"text":"claim","chunk_ids":["chunk"]}],"no_evidence":false}',
        retrieved_chunks=[chunk],
        context=context,
    )
    assert result.citation_validation_failed
    assert result.claims == []
    assert result.citation_validation_status == "revision_mismatch"
