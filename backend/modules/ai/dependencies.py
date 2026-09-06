from fastapi import Depends, Request

from backend.api.deps.auth import get_current_user
from backend.core.config import settings
from backend.core.rate_limit import check_rate_limit
from backend.modules.identity_access.models import User


async def enforce_ai_generation_rate_limit(
    request: Request,
    current_user: User = Depends(get_current_user),
) -> None:
    if settings.AI_RATE_LIMIT_REQUESTS <= 0:
        return
    client_ip = request.client.host if request.client else "unknown"
    await check_rate_limit(
        f"rate_limit:ai:{current_user.id}:{client_ip}",
        settings.AI_RATE_LIMIT_REQUESTS,
        settings.AI_RATE_LIMIT_WINDOW_SECONDS,
    )
