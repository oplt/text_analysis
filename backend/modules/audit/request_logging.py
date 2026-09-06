from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from backend.modules.audit.repository import AuditRepository


async def log_request_audit_event(
    db: AsyncSession,
    request: Request,
    *,
    action: str,
    actor_user_id: str,
    resource_type: str,
    resource_id: str | None = None,
    metadata: dict | None = None,
) -> None:
    """Record an audit event with the available HTTP client context."""
    await AuditRepository(db).log(
        action=action,
        user_id=actor_user_id,
        resource_type=resource_type,
        resource_id=resource_id,
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
        metadata=metadata,
    )
