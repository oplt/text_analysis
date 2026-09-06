import hashlib
from datetime import datetime

from fastapi import HTTPException

from backend.core.cache import (
    PLATFORM_FEATURE_FLAGS_CACHE_KEY,
    cache_get_or_load_json,
    invalidate_platform_caches,
)
from backend.core.config import settings
from backend.modules.identity_access.models import User
from backend.modules.platform.config_service import PlatformConfigService
from backend.modules.platform.models import FeatureFlag
from backend.modules.platform.schemas import EffectiveFeatureFlagResponse


class FeatureFlagService(PlatformConfigService):
    async def list_feature_flags(self) -> list[FeatureFlag]:
        cached = await cache_get_or_load_json(
            PLATFORM_FEATURE_FLAGS_CACHE_KEY,
            ttl_seconds=settings.CACHE_PLATFORM_TTL_SECONDS,
            loader=self._load_feature_flags_for_cache,
        )
        return [self._feature_flag_from_cache(item) for item in cached]

    async def _load_feature_flags_for_cache(self) -> list[dict]:
        flags = await self.repo.list_feature_flags()
        return [self._feature_flag_to_cache(flag) for flag in flags]

    async def list_effective_feature_flags_for_user(
        self, user: User
    ) -> list[EffectiveFeatureFlagResponse]:
        await self.ensure_module_enabled("feature_flags")
        flags = await self.list_feature_flags()
        metadata = await self.get_platform_metadata()
        enabled_modules = set(metadata.enabled_modules)
        return [
            EffectiveFeatureFlagResponse(
                id=flag.id,
                key=flag.key,
                name=flag.name,
                description=flag.description,
                module_key=flag.module_key,
                is_enabled=flag.is_enabled,
                rollout_percentage=flag.rollout_percentage,
                updated_at=flag.updated_at,
                effective_enabled=self._is_flag_effective(flag, user.id, enabled_modules),
            )
            for flag in flags
        ]

    async def create_feature_flag(self, payload: dict) -> FeatureFlag:
        if await self.repo.get_feature_flag_by_key(payload["key"]) is not None:
            raise HTTPException(
                status_code=409, detail="A feature flag with this key already exists"
            )

        flag = await self.repo.create_feature_flag(**payload)
        await self.db.commit()
        await self.db.refresh(flag)
        await invalidate_platform_caches()
        return flag

    async def update_feature_flag(self, feature_flag_id: str, payload: dict) -> FeatureFlag:
        flag = await self.repo.get_feature_flag_by_id(feature_flag_id)
        if not flag:
            raise HTTPException(status_code=404, detail="Feature flag not found")
        for field, value in payload.items():
            setattr(flag, field, value)
        await self.db.commit()
        await self.db.refresh(flag)
        await invalidate_platform_caches()
        return flag

    @staticmethod
    def _feature_flag_to_cache(flag: FeatureFlag) -> dict:
        return {
            "id": flag.id,
            "key": flag.key,
            "name": flag.name,
            "description": flag.description,
            "module_key": flag.module_key,
            "is_enabled": flag.is_enabled,
            "rollout_percentage": flag.rollout_percentage,
            "updated_at": flag.updated_at.isoformat(),
        }

    @staticmethod
    def _feature_flag_from_cache(payload: dict) -> FeatureFlag:
        data = dict(payload)
        updated_at = data.get("updated_at")
        if isinstance(updated_at, str):
            data["updated_at"] = datetime.fromisoformat(updated_at)
        return FeatureFlag(**data)

    @staticmethod
    def _is_flag_effective(flag: FeatureFlag, user_id: str, enabled_modules: set[str]) -> bool:
        if not flag.is_enabled:
            return False
        if flag.module_key and flag.module_key not in enabled_modules:
            return False
        if flag.rollout_percentage >= 100:
            return True
        if flag.rollout_percentage <= 0:
            return False
        digest = hashlib.sha256(f"{flag.key}:{user_id}".encode()).hexdigest()
        bucket = int(digest[:8], 16) % 100
        return bucket < flag.rollout_percentage
