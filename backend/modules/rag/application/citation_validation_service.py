"""Server-side claim-level citation validation (never trust model IDs)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

from backend.modules.rag.domain.models import ClaimCitation, Citation, RetrievedChunk

_JSON_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)

CitationValidationStatus = Literal["valid", "partial", "unstructured", "invalid"]


@dataclass(slots=True)
class ValidatedAnswer:
    answer: str
    claims: list[ClaimCitation]
    citations: list[Citation]
    citation_validation_failed: bool = False
    citation_validation_status: CitationValidationStatus = "valid"


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
            snippet=c.content[:400],
            page_number=c.page_number,
            chunk_index=c.chunk_index,
            section_heading=(c.metadata or {}).get("section_heading"),
            used_in_answer=False,
            char_start=(c.metadata or {}).get("char_start"),
            char_end=(c.metadata or {}).get("char_end"),
            source_span_ids=(c.metadata or {}).get("source_span_ids"),
        )
        for c in retrieved_chunks
    ]


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

        payload = _parse_structured_payload(raw_output)
        if payload is None:
            # Unstructured fallback must NOT be reported as citation-validated success.
            return ValidatedAnswer(
                answer=raw_output,
                claims=[],
                citations=_unused_citations(retrieved_chunks),
                citation_validation_failed=True,
                citation_validation_status="unstructured",
            )

        answer = str(payload.get("answer") or "").strip()
        raw_claims = payload.get("claims") or []
        if not isinstance(raw_claims, list):
            raw_claims = []

        used_chunk_ids: list[str] = []
        claims: list[ClaimCitation] = []
        rejected_any = False
        accepted_any = False

        for item in raw_claims:
            if not isinstance(item, dict):
                rejected_any = True
                continue
            claim_text = str(item.get("text") or "").strip()
            chunk_ids = item.get("chunk_ids") or []
            if not isinstance(chunk_ids, list):
                rejected_any = True
                continue
            valid_ids: list[str] = []
            for chunk_id in chunk_ids:
                cid = str(chunk_id)
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
            if claim_text and valid_ids:
                claims.append(ClaimCitation(text=claim_text, chunk_ids=valid_ids))
            elif claim_text and not valid_ids and chunk_ids:
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
                    snippet=chunk.content[:400],
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    citation_number=index,
                    section_heading=(chunk.metadata or {}).get("section_heading"),
                    used_in_answer=True,
                    char_start=(chunk.metadata or {}).get("char_start"),
                    char_end=(chunk.metadata or {}).get("char_end"),
                    source_span_ids=(chunk.metadata or {}).get("source_span_ids"),
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
                    snippet=chunk.content[:400],
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    citation_number=None,
                    section_heading=(chunk.metadata or {}).get("section_heading"),
                    used_in_answer=False,
                    char_start=(chunk.metadata or {}).get("char_start"),
                    char_end=(chunk.metadata or {}).get("char_end"),
                    source_span_ids=(chunk.metadata or {}).get("source_span_ids"),
                )
            )

        if rejected_any and not accepted_any and raw_claims:
            status: CitationValidationStatus = "invalid"
        elif rejected_any:
            status = "partial"
        else:
            status = "valid"

        return ValidatedAnswer(
            answer=answer or raw_output,
            claims=claims,
            citations=citations,
            citation_validation_failed=status in {"invalid", "unstructured", "partial"},
            citation_validation_status=status,
        )
