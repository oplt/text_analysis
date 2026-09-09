"""Stratified / seeded annotation sampling (prompt.txt §24).

Builds a reproducible ``SamplingPlan`` over a corpus's text units and
composes with :mod:`assignment_planning` for overlap/shared/disjoint
distribution of the sampled pool across annotators.

Stratification is fully generic: callers choose any combination of
supported document metadata fields to stratify on (built-in columns such as
``organization``/``country``/``publication_year``, plus any custom key
stored in a document's free-form ``metadata_json``). There are no hardcoded
geographic or organizational dimensions here — a caller is just as free to
stratify by ``field_1`` x ``field_2`` as by ``country`` x ``organization``.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from backend.modules.text_research.application.access import ResearchAccessMixin
from backend.modules.text_research.domain.models import CorpusDocument, loads
from backend.modules.text_research.infrastructure.sampling import (
    SAMPLING_LEVEL_UNIT,
    STRATUM_MODE_PROPORTIONAL,
    SamplingItem,
    build_sampling_plan,
)

# Built-in CorpusDocument columns that are safe/meaningful to stratify on.
# Any other field name is looked up in the document's metadata_json instead,
# so users are never limited to this list.
_DOCUMENT_METADATA_COLUMNS: tuple[str, ...] = (
    "organization",
    "organization_type",
    "publication_year",
    "publication_type",
    "country",
    "region",
    "cultural_sphere",
    "language",
    "education_level",
)


def document_metadata_fields(document: CorpusDocument) -> dict[str, Any]:
    """Flatten a document's built-in columns + custom metadata_json into one
    dict usable for generic stratification by any field name."""
    metadata: dict[str, Any] = {
        field_name: getattr(document, field_name, None)
        for field_name in _DOCUMENT_METADATA_COLUMNS
    }
    extra = loads(document.metadata_json, default=None)
    if isinstance(extra, dict):
        for key, value in extra.items():
            # Built-in columns take precedence over same-named custom keys.
            metadata.setdefault(key, value)
    return metadata


class SamplingService(ResearchAccessMixin):
    """Builds `SamplingPlan`s for annotation task assignment."""

    async def build_corpus_sampling_plan(
        self,
        corpus_id: str,
        *,
        user_id: str,
        unit_type: str,
        sample_size: int,
        candidate_unit_ids: list[str] | None = None,
        random_seed: int | None = None,
        stratify_by: list[str] | None = None,
        stratum_mode: str = STRATUM_MODE_PROPORTIONAL,
        sampling_level: str = SAMPLING_LEVEL_UNIT,
        max_units_per_document: int | None = None,
    ) -> dict[str, Any]:
        """Return a `SamplingPlan` dict for the given corpus.

        When ``candidate_unit_ids`` is provided, sampling is restricted to
        that pool (e.g. units not already fully assigned); otherwise every
        ``unit_type`` unit in the corpus is eligible.
        """
        await self.get_corpus_or_404(corpus_id, user_id=user_id)

        units = await self.repo.list_text_units_for_corpus(corpus_id, unit_type=unit_type)
        if candidate_unit_ids is not None:
            allowed = set(candidate_unit_ids)
            units = [unit for unit in units if unit.id in allowed]
        if not units:
            raise HTTPException(
                status_code=422, detail="No candidate units available for sampling"
            )

        document_ids = list(dict.fromkeys(unit.corpus_document_id for unit in units))
        documents = await self.repo.list_documents_by_ids(document_ids)
        metadata_by_document = {doc.id: document_metadata_fields(doc) for doc in documents}

        items = [
            SamplingItem(
                id=unit.id,
                document_id=unit.corpus_document_id,
                metadata=metadata_by_document.get(unit.corpus_document_id, {}),
            )
            for unit in units
        ]

        try:
            plan = build_sampling_plan(
                items,
                sample_size=sample_size,
                random_seed=random_seed,
                stratify_by=stratify_by,
                stratum_mode=stratum_mode,
                sampling_level=sampling_level,
                max_units_per_document=max_units_per_document,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        plan["sampling_config"]["unit_type"] = unit_type
        plan["sampling_config"]["corpus_id"] = corpus_id
        return plan
