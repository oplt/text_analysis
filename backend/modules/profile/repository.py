from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.modules.profile.models import UserProfile


class ProfileRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_by_user_id(self, user_id: str) -> UserProfile | None:
        result = await self.db.execute(
            select(UserProfile).where(UserProfile.user_id == user_id)
        )
        return result.scalar_one_or_none()

    async def get_or_create(self, user_id: str) -> UserProfile:
        profile = await self.get_by_user_id(user_id)
        if not profile:
            profile = UserProfile(user_id=user_id)
            self.db.add(profile)
            await self.db.flush()
        return profile

    async def update(self, profile: UserProfile, **kwargs) -> UserProfile:
        for key, value in kwargs.items():
            if value is not None or key in kwargs:
                setattr(profile, key, value)
        await self.db.flush()
        return profile
