from collections.abc import AsyncGenerator

from backend.db.session import SessionLocal


async def get_db() -> AsyncGenerator:
    async with SessionLocal() as session:
        yield session