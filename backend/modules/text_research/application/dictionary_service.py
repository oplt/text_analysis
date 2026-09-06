"""Dictionary (word-list) CRUD and versioning.

Dictionary analysis results (`QuantitativeAnalysisService.dictionary`) are
always tied to a specific `DictionaryDefinition.id`/`version` for provenance.
"""

from __future__ import annotations

import json

from backend.modules.text_research.application.access import ResearchAccessMixin
from backend.modules.text_research.domain.models import DictionaryDefinition


class DictionaryService(ResearchAccessMixin):
    async def create_dictionary(
        self,
        *,
        project_id: str,
        user_id: str,
        name: str,
        terms: list[str],
        description: str | None = None,
        version: str = "1.0",
    ) -> DictionaryDefinition:
        await self.ensure_project_access(user_id=user_id, project_id=project_id)
        dictionary = await self.repo.create_dictionary(
            DictionaryDefinition(
                project_id=project_id,
                name=name,
                version=version,
                description=description,
                terms_json=json.dumps(terms, ensure_ascii=True),
                created_by=user_id,
            )
        )
        await self.db.commit()
        return dictionary

    async def list_dictionaries(self, *, project_id: str, user_id: str) -> list[DictionaryDefinition]:
        await self.ensure_project_access(user_id=user_id, project_id=project_id)
        return await self.repo.list_dictionaries(project_id)

    async def get_dictionary(self, dictionary_id: str, *, user_id: str) -> DictionaryDefinition:
        return await self.get_dictionary_or_404(dictionary_id, user_id=user_id)

    async def update_dictionary(
        self,
        dictionary_id: str,
        *,
        user_id: str,
        name: str | None = None,
        description: str | None = None,
        terms: list[str] | None = None,
    ) -> DictionaryDefinition:
        dictionary = await self.get_dictionary_or_404(dictionary_id, user_id=user_id)
        fields: dict = {}
        if name is not None:
            fields["name"] = name
        if description is not None:
            fields["description"] = description
        if terms is not None:
            fields["terms_json"] = json.dumps(terms, ensure_ascii=True)
        updated = await self.repo.update_dictionary(dictionary, **fields)
        await self.db.commit()
        return updated

    async def create_version(
        self,
        dictionary_id: str,
        *,
        user_id: str,
        new_version: str,
        terms: list[str] | None = None,
    ) -> DictionaryDefinition:
        source = await self.get_dictionary_or_404(dictionary_id, user_id=user_id)
        new_terms = terms if terms is not None else json.loads(source.terms_json)
        new_dictionary = await self.repo.create_dictionary(
            DictionaryDefinition(
                project_id=source.project_id,
                name=source.name,
                version=new_version,
                description=source.description,
                terms_json=json.dumps(new_terms, ensure_ascii=True),
                created_by=user_id,
            )
        )
        await self.db.commit()
        return new_dictionary

    @staticmethod
    def get_terms(dictionary: DictionaryDefinition) -> list[str]:
        return json.loads(dictionary.terms_json)
