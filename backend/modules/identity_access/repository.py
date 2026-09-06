from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.modules.identity_access.models import RefreshSession, User


class IdentityRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_user_by_email(self, email: str) -> User | None:
        result = await self.db.execute(select(User).where(User.email == email))
        return result.scalar_one_or_none()

    async def get_user_by_id(self, user_id: str) -> User | None:
        result = await self.db.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()

    async def create_user(
        self,
        email: str,
        password_hash: str,
        full_name: str | None,
        is_admin: bool = False,
        is_verified: bool = False,
    ) -> User:
        user = User(
            email=email,
            password_hash=password_hash,
            full_name=full_name,
            is_admin=is_admin,
            is_verified=is_verified,
        )
        self.db.add(user)
        await self.db.flush()
        return user

    async def create_refresh_session(
            self, user_id: str, token_hash: str, expires_at: datetime
    ) -> RefreshSession:
        session = RefreshSession(
            user_id=user_id,
            token_hash=token_hash,
            expires_at=expires_at,
        )
        self.db.add(session)
        await self.db.flush()
        return session

    async def get_refresh_session_by_hash(self, token_hash: str) -> RefreshSession | None:
        result = await self.db.execute(
            select(RefreshSession).where(RefreshSession.token_hash == token_hash)
        )
        return result.scalar_one_or_none()

    async def revoke_refresh_session(self, session: RefreshSession) -> None:
        session.is_revoked = True
        await self.db.flush()

    async def revoke_all_refresh_sessions_for_user(self, user_id: str) -> None:
        sessions = await self.list_active_sessions(user_id, limit=None)
        for session in sessions:
            session.is_revoked = True
        await self.db.flush()

    async def list_active_sessions(
        self, user_id: str, *, limit: int | None = 50
    ) -> list[RefreshSession]:
        now = datetime.now(UTC)
        stmt = (
            select(RefreshSession)
            .where(
                RefreshSession.user_id == user_id,
                RefreshSession.is_revoked.is_(False),
                RefreshSession.expires_at > now,
            )
            .order_by(RefreshSession.created_at.desc())
        )
        if limit is not None:
            stmt = stmt.limit(limit)
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def get_session_by_id(self, session_id: str) -> RefreshSession | None:
        result = await self.db.execute(
            select(RefreshSession).where(RefreshSession.id == session_id)
        )
        return result.scalar_one_or_none()

    async def get_active_session_by_id(self, session_id: str) -> RefreshSession | None:
        session = await self.get_session_by_id(session_id)
        if not session or session.is_revoked:
            return None
        if session.expires_at <= datetime.now(UTC):
            return None
        return session
