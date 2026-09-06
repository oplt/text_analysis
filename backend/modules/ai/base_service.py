from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from backend.modules.ai.providers import AiProviderRegistry
from backend.modules.ai.repository import AiRepository


class AiBaseService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.repo = AiRepository(db)
        self.providers = AiProviderRegistry()
