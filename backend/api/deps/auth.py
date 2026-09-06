import logging

from fastapi import Cookie, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps.db import get_db
from backend.core.config import settings
from backend.core.security import decode_token
from backend.modules.identity_access.models import User
from backend.modules.identity_access.repository import IdentityRepository

logger = logging.getLogger("backend.authz")


async def _get_authenticated_user(access_token: str | None, db: AsyncSession) -> User:
    try:
        if not access_token:
            raise HTTPException(status_code=401, detail="Authentication required")
        payload = decode_token(access_token)
        user_id = payload["sub"]
        session_id = payload["sid"]
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=401, detail="Invalid token") from exc

    repo = IdentityRepository(db)
    user = await repo.get_user_by_id(user_id)
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found")
    if payload.get("type") != "access":
        raise HTTPException(status_code=401, detail="Invalid token type")
    session = await repo.get_active_session_by_id(session_id)
    if not session or session.user_id != user.id:
        raise HTTPException(status_code=401, detail="Session is no longer valid")

    return user


async def get_current_user(
    access_token: str | None = Cookie(default=None, alias=settings.ACCESS_COOKIE_NAME),
    db: AsyncSession = Depends(get_db),
) -> User:
    user = await _get_authenticated_user(access_token, db)
    if not user.is_verified:
        logger.warning("authorization_failed action=unverified_access user_id=%s", user.id)
        raise HTTPException(status_code=403, detail="Verify your email before accessing the app")
    return user


async def get_authenticated_user(
    access_token: str | None = Cookie(default=None, alias=settings.ACCESS_COOKIE_NAME),
    db: AsyncSession = Depends(get_db),
) -> User:
    return await _get_authenticated_user(access_token, db)
