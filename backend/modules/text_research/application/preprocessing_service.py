"""CRUD for `PreprocessingProfile` records used by every quantitative /
topic-model / classifier run for reproducibility."""

from __future__ import annotations

import json

from backend.modules.text_research.application.access import ResearchAccessMixin
from backend.modules.text_research.domain.models import PreprocessingProfile
from backend.modules.text_research.infrastructure.preprocessing import PreprocessingConfig


class PreprocessingProfileService(ResearchAccessMixin):
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
        resolved = PreprocessingConfig.from_dict(config).to_dict()
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
            fields["config_json"] = json.dumps(
                PreprocessingConfig.from_dict(config).to_dict(), ensure_ascii=True
            )
        updated = await self.repo.update_preprocessing_profile(profile, **fields)
        await self.db.commit()
        return updated

    async def delete_profile(self, profile_id: str, *, user_id: str) -> None:
        profile = await self.get_preprocessing_profile_or_404(profile_id, user_id=user_id)
        await self.repo.delete_preprocessing_profile(profile)
        await self.db.commit()

    @staticmethod
    def resolve_config(profile: PreprocessingProfile | None) -> PreprocessingConfig:
        if profile is None:
            return PreprocessingConfig()
        return PreprocessingConfig.from_dict(json.loads(profile.config_json))
