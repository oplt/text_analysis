"""CRUD for `PreprocessingProfile` records used by every quantitative /
topic-model / classifier run for reproducibility."""

from __future__ import annotations

import json

from fastapi import HTTPException

from backend.modules.text_research.application.access import ResearchAccessMixin
from backend.modules.text_research.domain.models import PreprocessingProfile
from backend.modules.text_research.infrastructure.preprocessing import (
    PreprocessingConfig,
    preview_preprocessing,
)


class PreprocessingProfileService(ResearchAccessMixin):
    @staticmethod
    def _resolved_config_dict(config: dict | None) -> dict:
        try:
            return PreprocessingConfig.from_dict(config).to_dict()
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    async def create_profile(
        self,
        *,
        project_id: str,
        user_id: str,
        name: str,
        description: str | None = None,
        config: dict | None = None,
    ) -> PreprocessingProfile:
        await self.ensure_project_access(user_id=user_id, project_id=project_id)
        resolved = self._resolved_config_dict(config)
        profile = await self.repo.create_preprocessing_profile(
            project_id=project_id,
            name=name,
            description=description,
            config_json=json.dumps(resolved, ensure_ascii=True),
            created_by=user_id,
        )
        await self.db.commit()
        return profile

    async def list_profiles(self, *, project_id: str, user_id: str) -> list[PreprocessingProfile]:
        await self.ensure_project_access(user_id=user_id, project_id=project_id)
        return await self.repo.list_preprocessing_profiles(project_id)

    async def get_profile(self, profile_id: str, *, user_id: str) -> PreprocessingProfile:
        return await self.get_preprocessing_profile_or_404(profile_id, user_id=user_id)

    async def update_profile(
        self,
        profile_id: str,
        *,
        user_id: str,
        name: str | None = None,
        description: str | None = None,
        config: dict | None = None,
    ) -> PreprocessingProfile:
        profile = await self.get_preprocessing_profile_or_404(profile_id, user_id=user_id)
        fields: dict = {}
        if name is not None:
            fields["name"] = name
        if description is not None:
            fields["description"] = description
        if config is not None:
            fields["config_json"] = json.dumps(self._resolved_config_dict(config), ensure_ascii=True)
        updated = await self.repo.update_preprocessing_profile(profile, **fields)
        await self.db.commit()
        return updated

    async def delete_profile(self, profile_id: str, *, user_id: str) -> None:
        profile = await self.get_preprocessing_profile_or_404(profile_id, user_id=user_id)
        await self.repo.delete_preprocessing_profile(profile)
        await self.db.commit()

    async def preview(
        self,
        *,
        user_id: str,
        project_id: str | None = None,
        corpus_id: str | None = None,
        unit_type: str | None = None,
        texts: list[str] | None = None,
        config: dict | None = None,
        preprocessing_profile_id: str | None = None,
        sample_size: int = 5,
    ) -> dict:
        resolved_config: dict | None = config
        profile_name: str | None = None
        profile_updated_at = None

        if preprocessing_profile_id:
            profile = await self.get_preprocessing_profile_or_404(
                preprocessing_profile_id, user_id=user_id
            )
            if project_id and profile.project_id != project_id:
                raise HTTPException(status_code=404, detail="Preprocessing profile not found")
            resolved_config = json.loads(profile.config_json)
            profile_name = profile.name
            profile_updated_at = profile.updated_at.isoformat()
            project_id = profile.project_id
        elif project_id:
            await self.ensure_project_access(user_id=user_id, project_id=project_id)

        sample_texts = [t for t in (texts or []) if t and t.strip()]
        if not sample_texts and corpus_id:
            corpus = await self.get_corpus_or_404(corpus_id, user_id=user_id)
            if project_id and corpus.project_id != project_id:
                raise HTTPException(status_code=404, detail="Corpus not found")
            units = await self.repo.list_text_units_for_corpus(
                corpus_id, unit_type=unit_type or "paragraph"
            )
            sample_texts = [unit.text for unit in units[: max(1, min(sample_size, 20))]]

        if not sample_texts:
            raise HTTPException(
                status_code=422,
                detail="Provide sample texts or a segmented corpus to preview preprocessing.",
            )

        try:
            preview = preview_preprocessing(sample_texts, resolved_config)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        preview["profile_name"] = profile_name
        preview["profile_updated_at"] = profile_updated_at
        return preview

    @staticmethod
    def resolve_config(profile: PreprocessingProfile | None) -> PreprocessingConfig:
        if profile is None:
            return PreprocessingConfig()
        data = json.loads(profile.config_json)
        return PreprocessingConfig.from_dict(data)
