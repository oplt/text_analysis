import logging

from fastapi import Depends, HTTPException

from backend.api.deps.auth import get_current_user
from backend.modules.identity_access.models import User

logger = logging.getLogger("backend.authz")


async def get_admin_user(current_user: User = Depends(get_current_user)) -> User:
    if not current_user.is_admin:
        logger.warning("authorization_failed action=admin_access user_id=%s", current_user.id)
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user
