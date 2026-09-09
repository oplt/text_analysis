from __future__ import annotations

import json
import re
from typing import Any

from fastapi import HTTPException

from backend.core.pagination import DEFAULT_PAGE_LIMIT, MAX_PAGE_LIMIT
from backend.modules.ai.base_service import AiBaseService
from backend.modules.ai.models import AiPromptTemplate, AiPromptVersion
from backend.modules.identity_access.models import User

PLACEHOLDER_PATTERN = re.compile(r"{{\\s*([a-zA-Z_][a-zA-Z0-9_]*)\\s*}}")


def _render_template(template: str, variables: dict[str, Any]) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        value = variables.get(key)
        if value is None:
            return ""
        if isinstance(value, dict | list):
            return json.dumps(value, ensure_ascii=True)
        return str(value)

    return PLACEHOLDER_PATTERN.sub(replace, template)


class AiPromptService(AiBaseService):
    async def list_prompt_templates(
        self,
        user: User,
        *,
        limit: int = DEFAULT_PAGE_LIMIT,
        offset: int = 0,
    ):
        return await self.repo.list_prompt_templates_for_user(user.id, limit=limit, offset=offset)

    async def create_prompt_template(
        self, user: User, key: str, name: str, description: str | None
    ):
        existing = await self.repo.get_prompt_template_by_key_for_user(user.id, key)
        if existing:
            raise HTTPException(
                status_code=409,
                detail="A prompt template with this key already exists",
            )
        template = await self.repo.create_prompt_template(
            user_id=user.id,
            key=key,
            name=name,
            description=description,
        )
        await self.db.commit()
        await self.db.refresh(template)
        return template

    async def update_prompt_template(self, user: User, template_id: str, updates: dict[str, Any]):
        template = await self.repo.get_prompt_template_for_user(user.id, template_id)
        if not template:
            raise HTTPException(status_code=404, detail="Prompt template not found")
        if "active_version_id" in updates and updates["active_version_id"]:
            version = await self.repo.get_prompt_version(updates["active_version_id"])
            if not version or version.prompt_template_id != template.id:
                raise HTTPException(
                    status_code=404,
                    detail="Prompt version not found for this template",
                )
            if not version.is_published:
                raise HTTPException(
                    status_code=422,
                    detail="Only published versions can be activated",
                )
        for field, value in updates.items():
            setattr(template, field, value)
        await self.db.commit()
        await self.db.refresh(template)
        return template

    async def create_prompt_version(self, user: User, template_id: str, payload: dict[str, Any]):
        template = await self.repo.get_prompt_template_for_user(user.id, template_id)
        if not template:
            raise HTTPException(status_code=404, detail="Prompt template not found")
        versions, _ = await self.repo.list_prompt_versions(template.id, limit=1, offset=0)
        next_version_number = (versions[0].version_number + 1) if versions else 1
        version = await self.repo.create_prompt_version(
            prompt_template_id=template.id,
            version_number=next_version_number,
            provider_key=payload["provider_key"],
            model_name=payload["model_name"],
            system_prompt=payload["system_prompt"],
            user_prompt_template=payload["user_prompt_template"],
            variable_definitions_json=[
                item.model_dump() for item in payload["variable_definitions"]
            ],
            response_format=payload["response_format"],
            temperature=payload["temperature"],
            rollout_percentage=payload["rollout_percentage"],
            is_published=payload["is_published"],
            input_cost_per_million=payload["input_cost_per_million"],
            output_cost_per_million=payload["output_cost_per_million"],
            created_by_user_id=user.id,
        )
        if template.active_version_id is None and version.is_published:
            template.active_version_id = version.id
        await self.db.commit()
        await self.db.refresh(version)
        await self.db.refresh(template)
        return version

    async def update_prompt_version(
        self, user: User, template_id: str, version_id: str, updates: dict[str, Any]
    ):
        template = await self.repo.get_prompt_template_for_user(user.id, template_id)
        if not template:
            raise HTTPException(status_code=404, detail="Prompt template not found")
        version = await self.repo.get_prompt_version(version_id)
        if not version or version.prompt_template_id != template.id:
            raise HTTPException(status_code=404, detail="Prompt version not found")
        for field, value in updates.items():
            if field == "variable_definitions":
                version.variable_definitions_json = [item.model_dump() for item in value]
            else:
                setattr(version, field, value)
        await self.db.commit()
        await self.db.refresh(version)
        return version

    async def list_prompt_versions(
        self,
        user: User,
        template_id: str,
        *,
        limit: int = DEFAULT_PAGE_LIMIT,
        offset: int = 0,
    ):
        template = await self.repo.get_prompt_template_for_user(user.id, template_id)
        if not template:
            raise HTTPException(status_code=404, detail="Prompt template not found")
        return await self.repo.list_prompt_versions(template.id, limit=limit, offset=offset)

    async def _resolve_prompt_version(
        self,
        user: User,
        *,
        prompt_template_key: str | None,
        prompt_version_id: str | None,
    ) -> tuple[AiPromptTemplate | None, AiPromptVersion]:
        if prompt_version_id:
            version = await self.repo.get_prompt_version(prompt_version_id)
            if not version:
                raise HTTPException(status_code=404, detail="Prompt version not found")
            template = await self.repo.get_prompt_template_for_user(
                user.id, version.prompt_template_id
            )
            if not template:
                raise HTTPException(status_code=404, detail="Prompt template not found")
            return template, version
        if not prompt_template_key:
            raise HTTPException(
                status_code=422,
                detail="prompt_template_key or prompt_version_id is required",
            )
        template = await self.repo.get_prompt_template_by_key_for_user(user.id, prompt_template_key)
        if not template:
            raise HTTPException(status_code=404, detail="Prompt template not found")
        versions, _ = await self.repo.list_prompt_versions(
            template.id, limit=MAX_PAGE_LIMIT, offset=0
        )
        version = None
        if template.active_version_id:
            version = next(
                (item for item in versions if item.id == template.active_version_id), None
            )
        if version is None:
            version = next((item for item in versions if item.is_published), None)
        if version is None and versions:
            version = versions[0]
        if version is None:
            raise HTTPException(status_code=422, detail="This prompt template has no versions yet")
        return template, version
