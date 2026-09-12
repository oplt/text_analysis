"""Authorization + evidence-revision binding for citation validation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class CitationValidationContext:
    """Scope and revision identity that citations must satisfy.

    Successful citations must be in the retrieved evidence set, within
    ``allowed_document_ids`` (when set), and bound to the exact frozen
    index revision for their document. Missing revision identity fails
    closed — never upgraded to the current document revision.
    """

    user_id: str
    project_id: str | None
    corpus_id: str | None = None
    evidence_revision_hash: str | None = None
    index_revision_ids: list[str] = field(default_factory=list)
    revision_by_document: dict[str, str] = field(default_factory=dict)
    allowed_document_ids: list[str] | None = None
    retrieval_trace_id: str | None = None
