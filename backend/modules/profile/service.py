from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.cache import cache_get_or_load_model
from backend.core.config import settings
from backend.lib.resource_cache import (
    invalidate_user_profile_cache,
    user_profile_cache_key,
)
from backend.modules.profile.models import UserProfile
from backend.modules.profile.repository import ProfileRepository
from backend.modules.profile.schemas import ProfileResponse
from backend.modules.profile.serializers import profile_to_response


class ProfileService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.repo = ProfileRepository(db)

    async def get_profile_response(self, user_id: str) -> ProfileResponse:
        async def loader() -> ProfileResponse:
            profile = await self.repo.get_or_create(user_id)
            return profile_to_response(profile)

        return await cache_get_or_load_model(
            user_profile_cache_key(user_id),
            ProfileResponse,
            ttl_seconds=settings.CACHE_USER_PROFILE_TTL_SECONDS,
            loader=loader,
        )

    async def get_profile(self, user_id: str) -> UserProfile:
        return await self.repo.get_or_create(user_id)

    async def update_profile(
        self,
        user_id: str,
        bio: str | None,
        location: str | None,
        website: str | None,
    ) -> UserProfile:
        profile = await self.repo.get_or_create(user_id)
        await self.repo.update(profile, bio=bio, location=location, website=website)
        await self.db.commit()
        await self.db.refresh(profile)
        await invalidate_user_profile_cache(user_id)
        return profile

    async def set_avatar_url(self, user_id: str, url: str | None) -> UserProfile:
        profile = await self.repo.get_or_create(user_id)
        profile.avatar_url = url
        if url is None:
            profile.avatar_storage_key = None
        await self.db.commit()
        await self.db.refresh(profile)
        await invalidate_user_profile_cache(user_id)
        return profile

    async def replace_avatar(
        self,
        user_id: str,
        avatar_url: str,
        storage_key: str,
    ) -> tuple[UserProfile, str | None]:
        profile = await self.repo.get_or_create(user_id)
        previous_key = profile.avatar_storage_key
        profile.avatar_url = avatar_url
        profile.avatar_storage_key = storage_key
        await self.db.commit()
        await self.db.refresh(profile)
        await invalidate_user_profile_cache(user_id)
        return profile, previous_key

    async def clear_avatar(self, user_id: str) -> str | None:
        profile = await self.repo.get_or_create(user_id)
        previous_key = profile.avatar_storage_key
        profile.avatar_url = None
        profile.avatar_storage_key = None
        await self.db.commit()
        await self.db.refresh(profile)
        await invalidate_user_profile_cache(user_id)
        return previous_key
