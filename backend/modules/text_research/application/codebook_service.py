"""Codebook and annotation-label management.

Codebooks are versioned. A frozen codebook version can never be mutated —
researchers must create a new version (optionally cloned from a frozen one)
to change label definitions after annotation has started against it.
"""

from __future__ import annotations

import json

from fastapi import HTTPException

from backend.modules.text_research.application.access import ResearchAccessMixin
from backend.modules.text_research.domain.exceptions import CodebookFrozenError
from backend.modules.text_research.domain.models import AnnotationLabel, Codebook

#: Seeded demo labels. Illustrative placeholders for the demo workflow — not
#: authoritative operational definitions. Every seeded label uses `is_placeholder=True`.
PLACEHOLDER_LABEL_DEFINITIONS: tuple[dict[str, str], ...] = (
    {
        "name": "Liberalism",
        "description": (
            "PLACEHOLDER definition for demo purposes: discourse emphasizing individual "
            "rights, free choice, and limited state intervention in education policy."
        ),
    },
    {
        "name": "Universalism",
        "description": (
            "PLACEHOLDER definition for demo purposes: discourse framing education as a "
            "universal right or good applicable to all, regardless of context."
        ),
    },
    {
        "name": "Individualism",
        "description": (
            "PLACEHOLDER definition for demo purposes: discourse foregrounding individual "
            "achievement, responsibility, or development over collective/group framing."
        ),
    },
    {
        "name": "Multiculturalism",
        "description": (
            "PLACEHOLDER definition for demo purposes: discourse recognizing or promoting "
            "cultural diversity and pluralism within education policy."
        ),
    },
)


class CodebookService(ResearchAccessMixin):
    async def create_codebook(
        self,
        *,
        project_id: str,
        user_id: str,
        name: str,
        description: str | None = None,
        version: str = "1.0",
        seed_placeholder_labels: bool = True,
    ) -> Codebook:
        await self.ensure_project_access(user_id=user_id, project_id=project_id)
        codebook = await self.repo.create_codebook(
            project_id=project_id,
            name=name,
            description=description,
            version=version,
            created_by=user_id,
        )
        if seed_placeholder_labels:
            for definition in PLACEHOLDER_LABEL_DEFINITIONS:
                await self.repo.create_label(
                    codebook_id=codebook.id,
                    name=definition["name"],
                    description=definition["description"],
                    is_placeholder=True,
                )
        await self.db.commit()
        return codebook

    async def list_codebooks(self, *, project_id: str, user_id: str) -> list[Codebook]:
        await self.ensure_project_access(user_id=user_id, project_id=project_id)
        return await self.repo.list_codebooks(project_id)

    async def get_codebook(self, codebook_id: str, *, user_id: str) -> Codebook:
        return await self.get_codebook_or_404(codebook_id, user_id=user_id)

    async def create_version(
        self, codebook_id: str, *, user_id: str, new_version: str
    ) -> Codebook:
        """Clone a codebook (frozen or not) into a new, mutable version."""
        source = await self.get_codebook_or_404(codebook_id, user_id=user_id)
        new_codebook = await self.repo.create_codebook(
            project_id=source.project_id,
            name=source.name,
            description=source.description,
            version=new_version,
            created_by=user_id,
        )
        for label in await self.repo.list_labels(source.id):
            await self.repo.create_label(
                codebook_id=new_codebook.id,
                name=label.name,
                description=label.description,
                inclusion_criteria=label.inclusion_criteria,
                exclusion_criteria=label.exclusion_criteria,
                positive_examples_json=label.positive_examples_json,
                negative_examples_json=label.negative_examples_json,
                is_placeholder=label.is_placeholder,
            )
        await self.db.commit()
        return new_codebook

    async def freeze_codebook(self, codebook_id: str, *, user_id: str) -> Codebook:
        codebook = await self.get_codebook_or_404(codebook_id, user_id=user_id)
        await self.repo.freeze_codebook(codebook)
        await self.db.commit()
        return codebook

    def _assert_not_frozen(self, codebook: Codebook) -> None:
        if codebook.is_frozen:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Codebook version is frozen and cannot be mutated. "
                    "Create a new version to change label definitions."
                ),
            )

    async def add_label(
        self,
        codebook_id: str,
        *,
        user_id: str,
        name: str,
        description: str | None = None,
        inclusion_criteria: str | None = None,
        exclusion_criteria: str | None = None,
        positive_examples: list[str] | None = None,
        negative_examples: list[str] | None = None,
        is_placeholder: bool = False,
    ) -> AnnotationLabel:
        codebook = await self.get_codebook_or_404(codebook_id, user_id=user_id)
        self._assert_not_frozen(codebook)
        label = await self.repo.create_label(
            codebook_id=codebook_id,
            name=name,
            description=description,
            inclusion_criteria=inclusion_criteria,
            exclusion_criteria=exclusion_criteria,
            positive_examples_json=json.dumps(positive_examples or [], ensure_ascii=True),
            negative_examples_json=json.dumps(negative_examples or [], ensure_ascii=True),
            is_placeholder=is_placeholder,
        )
        await self.db.commit()
        return label

    async def update_label(self, label_id: str, *, user_id: str, **fields) -> AnnotationLabel:
        label = await self.repo.get_label(label_id)
        if label is None:
            raise HTTPException(status_code=404, detail="Label not found")
        codebook = await self.get_codebook_or_404(label.codebook_id, user_id=user_id)
        self._assert_not_frozen(codebook)
        for key in ("positive_examples", "negative_examples"):
            if key in fields and fields[key] is not None:
                fields[f"{key}_json"] = json.dumps(fields.pop(key), ensure_ascii=True)
        updated = await self.repo.update_label(label, **fields)
        await self.db.commit()
        return updated

    async def list_labels(self, codebook_id: str, *, user_id: str) -> list[AnnotationLabel]:
        await self.get_codebook_or_404(codebook_id, user_id=user_id)
        return await self.repo.list_labels(codebook_id)


__all__ = ["CodebookService", "CodebookFrozenError", "PLACEHOLDER_LABEL_DEFINITIONS"]
