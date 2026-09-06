from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from backend.modules.settings.models import AppSetting


class SettingsRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def list_all(self) -> list[AppSetting]:
        result = await self.db.execute(select(AppSetting).order_by(AppSetting.key.asc()))
        return list(result.scalars().all())

    async def list_by_prefix(self, prefix: str) -> list[AppSetting]:
        result = await self.db.execute(
            select(AppSetting)
            .where(AppSetting.key.like(f"{prefix}%"))
            .order_by(AppSetting.key.asc())
        )
        return list(result.scalars().all())

    async def get_by_id(self, setting_id: str) -> AppSetting | None:
        result = await self.db.execute(select(AppSetting).where(AppSetting.id == setting_id))
        return result.scalar_one_or_none()

    async def get_by_key(self, key: str) -> AppSetting | None:
        result = await self.db.execute(select(AppSetting).where(AppSetting.key == key))
        return result.scalar_one_or_none()

    async def create(self, key: str, value: str, description: str | None) -> AppSetting:
        setting = AppSetting(key=key, value=value, description=description)
        self.db.add(setting)
        await self.db.flush()
        return setting

    def add_many(self, rows: list[dict[str, str]]) -> None:
        self.db.add_all([AppSetting(**row) for row in rows])

    async def upsert_defaults(self, rows: list[dict[str, str]]) -> int:
        if not rows:
            return 0
        statement = insert(AppSetting).values(rows).on_conflict_do_nothing(index_elements=["key"])
        result = await self.db.execute(statement)
        return int(result.rowcount or 0)

    async def delete(self, setting: AppSetting) -> None:
        await self.db.delete(setting)
        await self.db.flush()
