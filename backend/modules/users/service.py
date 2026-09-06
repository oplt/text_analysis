from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.config import settings
from backend.core.pagination import DEFAULT_PAGE_LIMIT
from backend.core.security import hash_password_async, verify_password_async
from backend.lib.resource_cache import (
    get_cached_model_list,
    invalidate_user_directory_cache,
    user_directory_cache_key,
    set_cached_model_list,
)
from backend.modules.identity_access.models import RefreshSession, User
from backend.modules.identity_access.repository import IdentityRepository
from backend.modules.users.repository import UsersRepository
from backend.modules.users.schemas import UserDirectoryResponse


class UsersService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.repo = UsersRepository(db)
        self.identity_repo = IdentityRepository(db)

    async def update_profile(self, user: User, full_name: str | None) -> User:
        updated = await self.repo.update_profile(user, full_name)
        await self.db.commit()
        await self.db.refresh(updated)
        await invalidate_user_directory_cache()
        return updated

    async def change_password(
        self, user: User, current_password: str, new_password: str
    ) -> None:
        if not await verify_password_async(current_password, user.password_hash):
            raise HTTPException(status_code=400, detail="Current password is incorrect")
        user.password_hash = await hash_password_async(new_password)
        await self.identity_repo.revoke_all_refresh_sessions_for_user(user.id)
        await self.db.flush()
        await self.db.commit()

    async def list_sessions(self, user: User) -> list[RefreshSession]:
        return await self.identity_repo.list_active_sessions(user.id)

    async def revoke_session(self, user: User, session_id: str) -> None:
        session = await self.identity_repo.get_session_by_id(session_id)
        if not session or session.user_id != user.id:
            raise HTTPException(status_code=404, detail="Session not found")
        await self.identity_repo.revoke_refresh_session(session)
        await self.db.commit()

    async def list_directory(
        self,
        *,
        limit: int = DEFAULT_PAGE_LIMIT,
        offset: int = 0,
    ) -> tuple[list[UserDirectoryResponse], int]:
        cache_key = user_directory_cache_key(limit, offset)
        cached = await get_cached_model_list(cache_key, UserDirectoryResponse)
        if cached is not None:
            return cached

        users, total = await self.repo.list_active_users(limit=limit, offset=offset)
        items = [
            UserDirectoryResponse(
                id=user.id,
                email=user.email,
                full_name=user.full_name,
            )
            for user in users
        ]
        await set_cached_model_list(
            cache_key,
            items,
            total=total,
            ttl_seconds=settings.CACHE_USER_DIRECTORY_TTL_SECONDS,
        )
        return items, total
