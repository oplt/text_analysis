"""Server-side claim-level citation validation (never trust model IDs)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

from backend.modules.rag.domain.models import Citation, ClaimCitation, RetrievedChunk
from pydantic import BaseModel, ConfigDict, Field, ValidationError

_JSON_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)

CitationValidationStatus = Literal[
    "valid",
    "no_evidence",
    "partial",
    "unstructured",
    "invalid",
    "missing_claims",
    "incomplete",
    "empty_answer",
    "answer_mismatch",
]


@dataclass(slots=True)
class ValidatedAnswer:
    answer: str
    claims: list[ClaimCitation]
    citations: list[Citation]
    citation_validation_failed: bool = False
    citation_validation_status: CitationValidationStatus = "valid"
    no_evidence: bool = False


class RagStructuredClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    chunk_ids: list[str] = Field(min_length=1)


class RagStructuredAnswer(BaseModel):
    """The only provider contract accepted for grounded RAG answers."""

    model_config = ConfigDict(extra="forbid")

    claims: list[RagStructuredClaim] = Field(default_factory=list)
    no_evidence: bool
    insufficient_evidence_reason: str | None = None


def _parse_structured_payload(raw: str) -> dict | None:
    text = (raw or "").strip()
    if not text:
        return None
    match = _JSON_FENCE.search(text)
    if match:
        text = match.group(1).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _unused_citations(retrieved_chunks: list[RetrievedChunk]) -> list[Citation]:
    return [
        Citation(
            document_id=c.document_id,
            chunk_id=c.chunk_id,
            filename=c.filename,
            score=c.score,
            snippet=(c.citation_content or c.content)[:400],
            page_number=c.page_number,
            chunk_index=c.chunk_index,
            section_heading=(c.metadata or {}).get("section_heading"),
            used_in_answer=False,
            char_start=(c.metadata or {}).get("char_start"),
            char_end=(c.metadata or {}).get("char_end"),
            source_span_ids=(c.metadata or {}).get("source_span_ids"),
            parent_context_id=c.parent_context_id,
            offset_coordinate_system=(c.metadata or {}).get("offset_coordinate_system"),
            offset_scope=(c.metadata or {}).get("offset_scope"),
        )
        for c in retrieved_chunks
    ]


def _answer_matches_claims(answer: str, claims: list[ClaimCitation]) -> bool:
    """Require each answer sentence to be represented by a validated claim."""
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", answer) if part.strip()]
    if not sentences:
        return False
    claim_texts = [
        normalized
        for claim in claims
        if (normalized := " ".join(re.findall(r"\w+", claim.text.casefold())))
    ]
    for sentence in sentences:
        normalized = " ".join(re.findall(r"\w+", sentence.casefold()))
        if not normalized:
            return False
        if any(
            normalized == claim or normalized in claim or claim in normalized
            for claim in claim_texts
        ):
            continue
        return False
    return True


class CitationValidationService:
    def validate(
        self,
        *,
        raw_output: str,
        retrieved_chunks: list[RetrievedChunk],
        allowed_document_ids: list[str] | None,
    ) -> ValidatedAnswer:
        by_id = {c.chunk_id: c for c in retrieved_chunks}
        allowed_docs = set(allowed_document_ids) if allowed_document_ids is not None else None

        raw_text = (raw_output or "").strip()
        payload = _parse_structured_payload(raw_text)
        if payload is None:
            # Unstructured fallback must NOT be reported as citation-validated success.
            return ValidatedAnswer(
                answer=raw_text,
                claims=[],
                citations=_unused_citations(retrieved_chunks),
                citation_validation_failed=True,
                citation_validation_status="empty_answer" if not raw_text else "unstructured",
            )

        # Older persisted/model responses included a free-form ``answer`` key.
        # Ignore it: the rendered answer is always reconstructed from validated
        # claims (or the explicit insufficient-evidence reason) below.
        payload = dict(payload)
        payload.pop("answer", None)
        payload.setdefault("no_evidence", False)
        payload.setdefault("insufficient_evidence_reason", None)
        try:
            structured = RagStructuredAnswer.model_validate(payload)
        except ValidationError:
            return ValidatedAnswer(
                answer="",
                claims=[],
                citations=_unused_citations(retrieved_chunks),
                citation_validation_failed=True,
                citation_validation_status=(
                    "incomplete" if "claims" in payload else "unstructured"
                ),
            )

        raw_claims = [claim.model_dump() for claim in structured.claims]
        claims_structurally_valid = True
        explicitly_no_evidence = structured.no_evidence
        if explicitly_no_evidence and not raw_claims:
            return ValidatedAnswer(
                answer=(
                    structured.insufficient_evidence_reason
                    or "The retrieved evidence is insufficient to answer this question."
                ),
                claims=[],
                citations=_unused_citations(retrieved_chunks),
                citation_validation_status="no_evidence",
                no_evidence=True,
            )

        used_chunk_ids: list[str] = []
        claims: list[ClaimCitation] = []
        rejected_any = False
        accepted_any = False
        incomplete_any = not claims_structurally_valid

        for item in raw_claims:
            if not isinstance(item, dict):
                rejected_any = True
                incomplete_any = True
                continue
            claim_text = item.get("text")
            chunk_ids = item.get("chunk_ids")
            if not isinstance(claim_text, str) or not claim_text.strip():
                rejected_any = True
                incomplete_any = True
                continue
            claim_text = claim_text.strip()
            if not isinstance(chunk_ids, list) or not chunk_ids:
                rejected_any = True
                incomplete_any = True
                continue
            valid_ids: list[str] = []
            for chunk_id in chunk_ids:
                if not isinstance(chunk_id, str) or not chunk_id:
                    rejected_any = True
                    continue
                cid = chunk_id
                chunk = by_id.get(cid)
                if chunk is None:
                    rejected_any = True
                    continue
                if allowed_docs is not None and chunk.document_id not in allowed_docs:
                    rejected_any = True
                    continue
                valid_ids.append(cid)
                accepted_any = True
                if cid not in used_chunk_ids:
                    used_chunk_ids.append(cid)
            if valid_ids:
                claims.append(ClaimCitation(text=claim_text, chunk_ids=valid_ids))
            else:
                rejected_any = True

        number_by_chunk: dict[str, int] = {}
        citations: list[Citation] = []
        for index, chunk_id in enumerate(used_chunk_ids, start=1):
            chunk = by_id[chunk_id]
            number_by_chunk[chunk_id] = index
            citations.append(
                Citation(
                    document_id=chunk.document_id,
                    chunk_id=chunk.chunk_id,
                    filename=chunk.filename,
                    score=chunk.score,
                    snippet=(chunk.citation_content or chunk.content)[:400],
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    citation_number=index,
                    section_heading=(chunk.metadata or {}).get("section_heading"),
                    used_in_answer=True,
                    char_start=(chunk.metadata or {}).get("char_start"),
                    char_end=(chunk.metadata or {}).get("char_end"),
                    source_span_ids=(chunk.metadata or {}).get("source_span_ids"),
                    parent_context_id=chunk.parent_context_id,
                    offset_coordinate_system=(chunk.metadata or {}).get("offset_coordinate_system"),
                    offset_scope=(chunk.metadata or {}).get("offset_scope"),
                )
            )

        for claim in claims:
            claim.citation_numbers = [number_by_chunk[cid] for cid in claim.chunk_ids]

        used_set = set(used_chunk_ids)
        for chunk in retrieved_chunks:
            if chunk.chunk_id in used_set:
                continue
            citations.append(
                Citation(
                    document_id=chunk.document_id,
                    chunk_id=chunk.chunk_id,
                    filename=chunk.filename,
                    score=chunk.score,
                    snippet=(chunk.citation_content or chunk.content)[:400],
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    citation_number=None,
                    section_heading=(chunk.metadata or {}).get("section_heading"),
                    used_in_answer=False,
                    char_start=(chunk.metadata or {}).get("char_start"),
                    char_end=(chunk.metadata or {}).get("char_end"),
                    source_span_ids=(chunk.metadata or {}).get("source_span_ids"),
                    parent_context_id=chunk.parent_context_id,
                    offset_coordinate_system=(chunk.metadata or {}).get("offset_coordinate_system"),
                    offset_scope=(chunk.metadata or {}).get("offset_scope"),
                )
            )

        answer = "\n\n".join(claim.text for claim in claims)
        if incomplete_any:
            status = "incomplete"
        elif not claims and not explicitly_no_evidence:
            status = "missing_claims" if not raw_claims else "invalid"
        elif rejected_any and not accepted_any:
            status: CitationValidationStatus = "invalid"
        elif rejected_any:
            status = "partial"
        else:
            status = "valid"

        return ValidatedAnswer(
            answer=answer,
            claims=claims,
            citations=citations,
            citation_validation_failed=status not in {"valid", "no_evidence"},
            citation_validation_status=status,
        )
