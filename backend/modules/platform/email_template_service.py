import re
from datetime import datetime

from fastapi import HTTPException

from backend.core.cache import (
    PLATFORM_EMAIL_TEMPLATES_CACHE_KEY,
    cache_get_or_load_json,
    invalidate_platform_email_template_cache,
)
from backend.core.config import settings
from backend.modules.platform.config_service import PlatformConfigService
from backend.modules.platform.models import EmailTemplate

PLACEHOLDER_PATTERN = re.compile(r"{{\\s*([a-zA-Z0-9_]+)\\s*}}")


class EmailTemplateService(PlatformConfigService):
    async def list_email_templates(self) -> list[EmailTemplate]:
        cached = await cache_get_or_load_json(
            PLATFORM_EMAIL_TEMPLATES_CACHE_KEY,
            ttl_seconds=settings.CACHE_SETTINGS_TTL_SECONDS,
            loader=self._load_email_templates_for_cache,
        )
        return [self._email_template_from_cache(item) for item in cached]

    async def _load_email_templates_for_cache(self) -> list[dict]:
        templates = await self.repo.list_email_templates()
        return [self._email_template_to_cache(template) for template in templates]

    async def create_email_template(self, payload: dict) -> EmailTemplate:
        if await self.repo.get_email_template_by_key(payload["key"]) is not None:
            raise HTTPException(
                status_code=409, detail="An email template with this key already exists"
            )
        template = await self.repo.create_email_template(**payload)
        await self.db.commit()
        await self.db.refresh(template)
        await invalidate_platform_email_template_cache()
        return template

    async def update_email_template(self, template_id: str, payload: dict) -> EmailTemplate:
        template = await self.repo.get_email_template_by_id(template_id)
        if not template:
            raise HTTPException(status_code=404, detail="Email template not found")
        for field, value in payload.items():
            setattr(template, field, value)
        await self.db.commit()
        await self.db.refresh(template)
        await invalidate_platform_email_template_cache()
        return template

    async def render_email_template(
        self,
        *,
        key: str,
        context: dict[str, str],
        fallback_subject: str,
        fallback_html: str,
        fallback_text: str | None = None,
    ) -> tuple[str, str, str | None]:
        template = await self.repo.get_email_template_by_key(key)
        if not template or not template.is_active:
            return fallback_subject, fallback_html, fallback_text
        return (
            self._render_template_string(template.subject_template, context),
            self._render_template_string(template.html_template, context),
            (
                self._render_template_string(template.text_template, context)
                if template.text_template
                else None
            ),
        )

    @staticmethod
    def _email_template_to_cache(template: EmailTemplate) -> dict:
        return {
            "id": template.id,
            "key": template.key,
            "name": template.name,
            "subject_template": template.subject_template,
            "html_template": template.html_template,
            "text_template": template.text_template,
            "is_active": template.is_active,
            "updated_at": template.updated_at.isoformat(),
        }

    @staticmethod
    def _email_template_from_cache(payload: dict) -> EmailTemplate:
        data = dict(payload)
        updated_at = data.get("updated_at")
        if isinstance(updated_at, str):
            data["updated_at"] = datetime.fromisoformat(updated_at)
        return EmailTemplate(**data)

    @staticmethod
    def _render_template_string(template: str, context: dict[str, str]) -> str:
        def _replace(match: re.Match[str]) -> str:
            key = match.group(1)
            return str(context.get(key, ""))

        return PLACEHOLDER_PATTERN.sub(_replace, template)
