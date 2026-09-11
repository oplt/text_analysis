"""Structured provenance helpers for corpus synthesis map/reduce runs."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from backend.modules.rag.domain.models import RetrievalOutcome

MAP_OPERATION = "deterministic_evidence_collection"
MAP_VERSION = "deterministic-evidence-map-v1"
REDUCTION_OPERATION = "structured_findings_reduction"
REDUCTION_VERSION = "structured-findings-reduce-v1"


def build_document_map_finding(
    *,
    question: str,
    rag_document_id: str,
    outcome: RetrievalOutcome,
) -> dict[str, Any]:
    """Build and validate a deterministic map output from one document's hits."""
    valid_chunks = [
        chunk
        for chunk in outcome.chunks
        if chunk.document_id == rag_document_id and bool(chunk.chunk_id)
    ]
    invalid_chunk_ids = [
        chunk.chunk_id
        for chunk in outcome.chunks
        if chunk.document_id != rag_document_id or not chunk.chunk_id
    ]
    passages = [
        {
            "chunk_id": chunk.chunk_id,
            "citation_content": (chunk.citation_content or chunk.content)[:400],
            "score": chunk.score,
            "page_number": chunk.page_number,
            "chunk_index": chunk.chunk_index,
            "char_start": (chunk.metadata or {}).get("char_start"),
            "char_end": (chunk.metadata or {}).get("char_end"),
            "source_span_ids": (chunk.metadata or {}).get("source_span_ids"),
            "parent_context_id": chunk.parent_context_id,
        }
        for chunk in valid_chunks
    ]
    missing = not valid_chunks
    main_finding = (
        "No relevant evidence was retrieved for this document."
        if missing
        else (
            f"Retrieved {len(valid_chunks)} evidence passage(s). "
            "Map stage records evidence only; it does not infer a document position."
        )
    )
    return {
        "map_operation": MAP_OPERATION,
        "map_version": MAP_VERSION,
        "research_question": question,
        "rag_document_id": rag_document_id,
        "filename": valid_chunks[0].filename if valid_chunks else None,
        "main_position": main_finding,
        "main_finding": main_finding,
        "finding": main_finding,
        "supporting_chunk_ids": [chunk.chunk_id for chunk in valid_chunks],
        "contradicting_or_qualifying_chunk_ids": [],
        "contradicting_or_qualifying_evidence": [],
        "agreements": [],
        "disagreements": [],
        "exceptions": [],
        "missing_evidence": missing,
        "missing_evidence_reason": (
            "No valid per-document retrieval hits" if missing else None
        ),
        "invalid_chunk_ids": invalid_chunk_ids,
        "chunk_reference_validation": "invalid" if invalid_chunk_ids else "valid",
        "passages": passages,
        "chunk_ids": [chunk.chunk_id for chunk in valid_chunks],
        "retrieval_trace_id": outcome.retrieval_trace_id,
    }


def _citation_dict(citation: Any) -> dict[str, Any]:
    return {
        "document_id": citation.document_id,
        "chunk_id": citation.chunk_id,
        "filename": citation.filename,
        "snippet": citation.snippet,
        "citation_number": citation.citation_number,
        "used_in_answer": citation.used_in_answer,
        "page_number": citation.page_number,
        "chunk_index": citation.chunk_index,
        "section_heading": citation.section_heading,
        "char_start": getattr(citation, "char_start", None),
        "char_end": getattr(citation, "char_end", None),
        "source_span_ids": getattr(citation, "source_span_ids", None),
        "parent_context_id": getattr(citation, "parent_context_id", None),
    }


def build_synthesis_provenance(
    *,
    original_question: str,
    evidence_revision_hash: str | None,
    corpus_id: str,
    project_id: str,
    user_id: str,
    documents_in_scope: list[str],
    documents_considered: list[str],
    omitted_documents: list[dict[str, str]],
    findings: list[dict[str, Any]],
    retrieval_trace_ids: list[str],
    reduction_prompt: str | None = None,
    answer: Any | None = None,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    """Return the complete auditable artifact embedded in synthesis results."""
    claims = [
        {
            "text": claim.text,
            "chunk_ids": claim.chunk_ids,
            "citation_numbers": claim.citation_numbers,
        }
        for claim in (getattr(answer, "claims", None) or [])
    ]
    return {
        "schema_version": "corpus-synthesis-provenance-v1",
        "original_research_question": original_question,
        "evidence_revision_hash": evidence_revision_hash,
        "corpus_id": corpus_id,
        "project_id": project_id,
        "created_by": user_id,
        "created_at": (created_at or datetime.now(UTC)).isoformat(),
        "documents_in_scope": list(documents_in_scope),
        "documents_considered": list(documents_considered),
        "documents_omitted": list(omitted_documents),
        "per_document_retrieval_trace_ids": [
            {
                "rag_document_id": finding["rag_document_id"],
                "retrieval_trace_id": finding.get("retrieval_trace_id"),
            }
            for finding in findings
        ],
        "retrieval_trace_ids": list(retrieval_trace_ids),
        "map_stage": {
            "operation": MAP_OPERATION,
            "version": MAP_VERSION,
            "model": None,
            "prompt": None,
            "outputs": findings,
        },
        "reduction_stage": {
            "operation": REDUCTION_OPERATION,
            "version": REDUCTION_VERSION,
            "prompt": reduction_prompt,
            "model": getattr(answer, "model_name", None),
            "prompt_template_id": getattr(answer, "prompt_template_id", None),
            "prompt_version_id": getattr(answer, "prompt_version_id", None),
        },
        "final_claims": claims,
        "final_citations": [
            _citation_dict(citation)
            for citation in (getattr(answer, "citations", None) or [])
        ],
        "citation_validation_status": getattr(answer, "citation_validation_status", None),
    }
