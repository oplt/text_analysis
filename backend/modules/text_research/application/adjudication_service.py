"""Disagreement listing and adjudication. Original annotations are never
mutated or deleted — adjudication only ever adds a separate gold record."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from backend.modules.text_research.application.access import ResearchAccessMixin
from backend.modules.text_research.domain.models import Adjudication
from backend.modules.text_research.infrastructure.reliability import disagreement_units


class AdjudicationService(ResearchAccessMixin):
    async def list_disagreements(
        self, corpus_id: str, *, user_id: str, codebook_id: str
    ) -> list[dict[str, Any]]:
        await self.get_corpus_or_404(corpus_id, user_id=user_id)
        codebook = await self.get_codebook_or_404(codebook_id, user_id=user_id)
        labels = await self.repo.list_labels(codebook_id)

        annotations = await self.repo.list_annotations_for_corpus(corpus_id)
        annotations = [a for a in annotations if a.codebook_version == codebook.version]

        by_label: dict[str, dict[str, dict[str, str]]] = {}
        for annotation in annotations:
            by_label.setdefault(annotation.label_id, {}).setdefault(annotation.annotator_id, {})[
                annotation.text_unit_id
            ] = annotation.value

        results: list[dict[str, Any]] = []
        for label in labels:
            values_by_coder = by_label.get(label.id, {})
            for row in disagreement_units(values_by_coder):
                existing = await self.repo.get_adjudication(
                    text_unit_id=row["text_unit_id"], label_id=label.id
                )
                if existing is not None:
                    continue
                results.append(
                    {
                        "text_unit_id": row["text_unit_id"],
                        "label_id": label.id,
                        "label_name": label.name,
                        "judgements": row["judgements"],
                    }
                )
        return results

    async def save_adjudication(
        self,
        *,
        user_id: str,
        text_unit_id: str,
        label_id: str,
        final_value: str,
        comment: str | None = None,
    ) -> Adjudication:
        await self.get_text_unit_or_404(text_unit_id, user_id=user_id)
        label = await self.repo.get_label(label_id)
        if label is None:
            raise HTTPException(status_code=404, detail="Label not found")

        adjudication = await self.repo.upsert_adjudication(
            text_unit_id=text_unit_id,
            label_id=label_id,
            final_value=final_value,
            adjudicator_id=user_id,
            comment=comment,
        )
        await self.db.commit()
        return adjudication

    async def list_adjudications(self, corpus_id: str, *, user_id: str) -> list[Adjudication]:
        await self.get_corpus_or_404(corpus_id, user_id=user_id)
        return await self.repo.list_adjudications_for_corpus(corpus_id)
