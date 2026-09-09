"""Dictionary (word-list) CRUD and versioning.

Dictionaries are always **user-defined**. Analysis results
(`QuantitativeAnalysisService.dictionary`) are tied to a specific
`DictionaryDefinition.id` / `version` for provenance.

No substantive research dictionaries are shipped as product defaults.
"""

from __future__ import annotations

import json
import re

from backend.modules.text_research.application.access import ResearchAccessMixin
from backend.modules.text_research.domain.models import DictionaryDefinition
from backend.modules.text_research.infrastructure.dictionary_matcher import (
    DictionarySpec,
    hierarchy_from_payload,
    parse_dictionary_payload,
    serialize_dictionary_payload,
)


def _bump_version(version: str) -> str:
    match = re.fullmatch(r"(\d+)(?:\.(\d+))?", str(version).strip())
    if not match:
        return f"{version}-next"
    major = int(match.group(1))
    minor = int(match.group(2) or 0)
    return f"{major}.{minor + 1}"


class DictionaryService(ResearchAccessMixin):
    async def create_dictionary(
        self,
        *,
        project_id: str,
        user_id: str,
        name: str,
        terms: list | dict | None = None,
        hierarchy: dict | None = None,
        exclusions: list | None = None,
        description: str | None = None,
        version: str = "1.0",
        language: str | None = None,
    ) -> DictionaryDefinition:
        await self.ensure_project_access(user_id=user_id, project_id=project_id)
        payload = self._build_payload(
            terms=terms,
            hierarchy=hierarchy,
            exclusions=exclusions,
            language=language,
            description=description,
            version=version,
            name=name,
        )
        spec = parse_dictionary_payload(
            payload, name=name, version=version, description=description, language=language
        )
        dictionary = await self.repo.create_dictionary(
            DictionaryDefinition(
                project_id=project_id,
                name=name,
                version=version,
                description=description,
                terms_json=serialize_dictionary_payload(
                    spec, hierarchy=hierarchy if hierarchy is not None else None
                ),
                created_by=user_id,
            )
        )
        await self.db.commit()
        return dictionary

    async def list_dictionaries(
        self, *, project_id: str, user_id: str
    ) -> list[DictionaryDefinition]:
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
        terms: list | dict | None = None,
        hierarchy: dict | None = None,
        exclusions: list | None = None,
        language: str | None = None,
        version: str | None = None,
    ) -> DictionaryDefinition:
        dictionary = await self.get_dictionary_or_404(dictionary_id, user_id=user_id)
        fields: dict = {}
        if name is not None:
            fields["name"] = name
        if description is not None:
            fields["description"] = description
        if version is not None:
            fields["version"] = version
        if any(v is not None for v in (terms, hierarchy, exclusions, language)):
            existing = json.loads(dictionary.terms_json)
            existing_spec = parse_dictionary_payload(
                existing,
                name=name or dictionary.name,
                version=version or dictionary.version,
                description=description if description is not None else dictionary.description,
                language=language,
            )
            if hierarchy is not None:
                payload = self._build_payload(
                    hierarchy=hierarchy,
                    exclusions=exclusions
                    if exclusions is not None
                    else [e.to_dict() for e in existing_spec.exclusions],
                    language=language if language is not None else existing_spec.language,
                    description=description if description is not None else dictionary.description,
                    version=version or dictionary.version,
                    name=name or dictionary.name,
                )
            elif terms is not None:
                payload = self._build_payload(
                    terms=terms,
                    exclusions=exclusions
                    if exclusions is not None
                    else [e.to_dict() for e in existing_spec.exclusions],
                    language=language if language is not None else existing_spec.language,
                    description=description if description is not None else dictionary.description,
                    version=version or dictionary.version,
                    name=name or dictionary.name,
                )
            else:
                # Language / exclusions only — keep existing terms/hierarchy.
                existing_hierarchy = hierarchy_from_payload(existing)
                if existing_hierarchy is not None:
                    payload = self._build_payload(
                        hierarchy=existing_hierarchy,
                        exclusions=exclusions
                        if exclusions is not None
                        else [e.to_dict() for e in existing_spec.exclusions],
                        language=language if language is not None else existing_spec.language,
                        description=description
                        if description is not None
                        else dictionary.description,
                        version=version or dictionary.version,
                        name=name or dictionary.name,
                    )
                else:
                    payload = self._build_payload(
                        terms=existing_spec.flattened_terms(),
                        exclusions=exclusions
                        if exclusions is not None
                        else [e.to_dict() for e in existing_spec.exclusions],
                        language=language if language is not None else existing_spec.language,
                        description=description
                        if description is not None
                        else dictionary.description,
                        version=version or dictionary.version,
                        name=name or dictionary.name,
                    )
            spec = parse_dictionary_payload(
                payload,
                name=name or dictionary.name,
                version=version or dictionary.version,
                description=description if description is not None else dictionary.description,
                language=language if language is not None else existing_spec.language,
            )
            fields["terms_json"] = serialize_dictionary_payload(
                spec,
                hierarchy=hierarchy if hierarchy is not None else hierarchy_from_payload(payload),
            )
        updated = await self.repo.update_dictionary(dictionary, **fields)
        await self.db.commit()
        return updated

    async def create_version(
        self,
        dictionary_id: str,
        *,
        user_id: str,
        new_version: str | None = None,
        terms: list | dict | None = None,
        hierarchy: dict | None = None,
    ) -> DictionaryDefinition:
        source = await self.get_dictionary_or_404(dictionary_id, user_id=user_id)
        version = new_version or _bump_version(source.version)
        source_payload = json.loads(source.terms_json)
        if terms is not None or hierarchy is not None:
            payload = self._build_payload(
                terms=terms,
                hierarchy=hierarchy,
                language=parse_dictionary_payload(source_payload).language,
                description=source.description,
                version=version,
                name=source.name,
            )
            spec = parse_dictionary_payload(
                payload,
                name=source.name,
                version=version,
                description=source.description,
            )
            terms_json = serialize_dictionary_payload(
                spec, hierarchy=hierarchy if hierarchy is not None else None
            )
        else:
            terms_json = source.terms_json
        new_dictionary = await self.repo.create_dictionary(
            DictionaryDefinition(
                project_id=source.project_id,
                name=source.name,
                version=version,
                description=source.description,
                terms_json=terms_json,
                created_by=user_id,
            )
        )
        await self.db.commit()
        return new_dictionary

    @staticmethod
    def get_terms(dictionary: DictionaryDefinition) -> list[str]:
        """Flattened leaf expressions (back-compat for simple UIs)."""
        return DictionaryService.get_spec(dictionary).flattened_terms()

    @staticmethod
    def get_spec(dictionary: DictionaryDefinition) -> DictionarySpec:
        return parse_dictionary_payload(
            json.loads(dictionary.terms_json),
            name=dictionary.name,
            version=dictionary.version,
            description=dictionary.description,
        )

    @staticmethod
    def get_hierarchy(dictionary: DictionaryDefinition) -> dict | None:
        return hierarchy_from_payload(json.loads(dictionary.terms_json))

    @staticmethod
    def _build_payload(
        *,
        terms: list | dict | None = None,
        hierarchy: dict | None = None,
        exclusions: list | None = None,
        language: str | None = None,
        description: str | None = None,
        version: str | None = None,
        name: str | None = None,
    ) -> list | dict:
        if hierarchy is not None:
            payload: dict = {"hierarchy": hierarchy, "source": "user"}
            if exclusions is not None:
                payload["exclusions"] = exclusions
            if language is not None:
                payload["language"] = language
            if description is not None:
                payload["description"] = description
            if version is not None:
                payload["version"] = version
            if name is not None:
                payload["name"] = name
            return payload
        if terms is None:
            raise ValueError("Provide terms or hierarchy — dictionaries are user-defined only")
        if isinstance(terms, dict):
            return terms
        if exclusions or language:
            payload = {"terms": terms, "source": "user"}
            if exclusions is not None:
                payload["exclusions"] = exclusions
            if language is not None:
                payload["language"] = language
            return payload
        return terms
