from collections.abc import AsyncGenerator
from time import monotonic

from backend.db.session import SessionLocal, observe_session_duration


async def get_db() -> AsyncGenerator:
    started_at = monotonic()
    async with SessionLocal() as session:
        try:
            yield session
        finally:
            observe_session_duration(role="api", started_at=started_at)
