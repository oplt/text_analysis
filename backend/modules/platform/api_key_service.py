import hashlib
import secrets
from datetime import UTC, datetime

from fastapi import HTTPException

from backend.core.pagination import DEFAULT_PAGE_LIMIT
from backend.modules.identity_access.models import User
from backend.modules.platform.config_service import PlatformConfigService
from backend.modules.platform.models import ApiKey


class ApiKeyService(PlatformConfigService):
    async def list_api_keys_for_user(
        self,
        user: User,
        *,
        limit: int = DEFAULT_PAGE_LIMIT,
        offset: int = 0,
    ) -> tuple[list[ApiKey], int]:
        await self.ensure_module_enabled("api_keys")
        return await self.repo.list_api_keys_for_user(user.id, limit=limit, offset=offset)

    async def create_api_key_for_user(self, user: User, name: str) -> tuple[ApiKey, str]:
        await self.ensure_module_enabled("api_keys")
        raw_token = f"gap_{secrets.token_urlsafe(32)}"
        api_key = await self.repo.create_api_key(
            user_id=user.id,
            name=name,
            key_prefix=raw_token[:12],
            key_hash=self._hash_secret(raw_token),
        )
        await self.db.commit()
        await self.db.refresh(api_key)
        return api_key, raw_token

    async def revoke_api_key_for_user(self, user: User, api_key_id: str) -> ApiKey:
        await self.ensure_module_enabled("api_keys")
        api_key = await self.repo.get_api_key_for_user(user.id, api_key_id)
        if not api_key:
            raise HTTPException(status_code=404, detail="API key not found")
        api_key.revoked_at = datetime.now(UTC)
        await self.db.commit()
        await self.db.refresh(api_key)
        return api_key

    @staticmethod
    def _hash_secret(raw_value: str) -> str:
        return hashlib.sha256(raw_value.encode("utf-8")).hexdigest()
