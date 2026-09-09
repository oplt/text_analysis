from __future__ import annotations

from typing import Protocol

from sqlalchemy.ext.asyncio import AsyncSession

from backend.modules.platform.config_service import PlatformConfigService
from backend.modules.platform.service import PlatformService


class PlatformReadPort(Protocol):
    async def get_app_name(self) -> str: ...

    async def render_email_template(
        self,
        *,
        key: str,
        context: dict[str, str],
        fallback_subject: str,
        fallback_html: str,
        fallback_text: str | None = None,
    ) -> tuple[str, str, str | None]: ...


class PlatformServiceReadPort:
    def __init__(self, db: AsyncSession):
        self._config = PlatformConfigService(db)
        self._platform = PlatformService(db)

    async def get_app_name(self) -> str:
        config = await self._config.get_platform_config()
        return config.app_name

    async def render_email_template(
        self,
        *,
        key: str,
        context: dict[str, str],
        fallback_subject: str,
        fallback_html: str,
        fallback_text: str | None = None,
    ) -> tuple[str, str, str | None]:
        return await self._platform.render_email_template(
            key=key,
            context=context,
            fallback_subject=fallback_subject,
            fallback_html=fallback_html,
            fallback_text=fallback_text,
        )
