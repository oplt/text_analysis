from backend.modules.rag.application.citation_validation_service import CitationValidationService
from backend.modules.rag.domain.citation_validation_context import CitationValidationContext
from backend.modules.rag.domain.models import RetrievedChunk


def _chunk(
    chunk_id: str,
    *,
    document_id: str = "document",
    revision: str | None = "rev-1",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        content="evidence",
        score=1.0,
        filename=f"{document_id}.txt",
        chunk_index=0,
        index_revision_id=revision,
    )


def _context(
    *,
    allowed_document_ids: list[str] | None = None,
    index_revision_ids: list[str] | None = None,
    revision_by_document: dict[str, str] | None = None,
    evidence_revision_hash: str | None = "evidence-hash",
) -> CitationValidationContext:
    return CitationValidationContext(
        user_id="user-1",
        project_id="project-1",
        corpus_id="corpus-1",
        evidence_revision_hash=evidence_revision_hash,
        index_revision_ids=list(index_revision_ids or []),
        revision_by_document=dict(revision_by_document or {}),
        allowed_document_ids=allowed_document_ids,
        retrieval_trace_id="trace-1",
    )


def _valid_payload(chunk_id: str = "chunk") -> str:
    return (
        '{"claims":[{"text":"claim","chunk_ids":["'
        + chunk_id
        + '"]}],"no_evidence":false}'
    )


def test_fabricated_chunk_not_in_retrieved_set_is_invalid():
    retrieved = [_chunk("real-chunk")]
    result = CitationValidationService().validate(
        raw_output=_valid_payload("fabricated-chunk"),
        retrieved_chunks=retrieved,
        context=_context(
            allowed_document_ids=["document"],
            index_revision_ids=["rev-1"],
            revision_by_document={"document": "rev-1"},
        ),
    )
    assert result.citation_validation_failed
    assert result.claims == []
    assert result.citation_validation_status == "invalid"


def test_wrong_revision_is_rejected():
    chunk = _chunk("chunk", revision="rev-new")
    result = CitationValidationService().validate(
        raw_output=_valid_payload("chunk"),
        retrieved_chunks=[chunk],
        context=_context(
            allowed_document_ids=["document"],
            index_revision_ids=["rev-frozen"],
            revision_by_document={"document": "rev-frozen"},
            # Hash present without consistent retrieved revisions → fail closed.
            evidence_revision_hash=None,
        ),
    )
    assert result.citation_validation_failed
    assert result.claims == []
    assert result.citation_validation_status == "revision_mismatch"


def test_out_of_scope_document_is_rejected():
    chunk = _chunk("chunk", document_id="other-doc", revision="rev-1")
    result = CitationValidationService().validate(
        raw_output=_valid_payload("chunk"),
        retrieved_chunks=[chunk],
        context=_context(
            allowed_document_ids=["allowed-doc"],
            index_revision_ids=["rev-1"],
            revision_by_document={"allowed-doc": "rev-1"},
            evidence_revision_hash=None,
        ),
    )
    assert result.citation_validation_failed
    assert result.claims == []
    assert result.citation_validation_status == "invalid"


def test_missing_revision_marked_invalid_not_silently_upgraded():
    chunk = _chunk("chunk", revision=None)
    result = CitationValidationService().validate(
        raw_output=_valid_payload("chunk"),
        retrieved_chunks=[chunk],
        context=_context(
            allowed_document_ids=["document"],
            index_revision_ids=["rev-1"],
            revision_by_document={"document": "rev-1"},
            evidence_revision_hash=None,
        ),
    )
    assert result.citation_validation_failed
    assert result.claims == []
    assert result.citation_validation_status == "revision_mismatch"
    # Snippet may remain for display, but must not be treated as validated use.
    assert all(not citation.used_in_answer for citation in result.citations)


def test_evidence_hash_with_inconsistent_retrieved_revisions_fails_closed():
    chunk = _chunk("chunk", revision=None)
    result = CitationValidationService().validate(
        raw_output=_valid_payload("chunk"),
        retrieved_chunks=[chunk],
        context=_context(
            allowed_document_ids=["document"],
            index_revision_ids=["rev-1"],
            revision_by_document={"document": "rev-1"},
            evidence_revision_hash="frozen-hash",
        ),
    )
    assert result.citation_validation_failed
    assert result.citation_validation_status == "revision_mismatch"
    assert result.claims == []


def test_exact_revision_and_acl_accepts_authorized_chunk():
    chunk = _chunk("chunk", revision="rev-1")
    result = CitationValidationService().validate(
        raw_output=_valid_payload("chunk"),
        retrieved_chunks=[chunk],
        context=_context(
            allowed_document_ids=["document"],
            index_revision_ids=["rev-1"],
            revision_by_document={"document": "rev-1"},
            evidence_revision_hash="frozen-hash",
        ),
    )
    assert not result.citation_validation_failed
    assert result.citation_validation_status == "valid"
    assert result.claims[0].chunk_ids == ["chunk"]
    assert result.citations[0].index_revision_id == "rev-1"
    assert result.citations[0].used_in_answer
