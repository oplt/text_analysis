from __future__ import annotations

import json
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.pagination import DEFAULT_PAGE_LIMIT, paginate_scalars
from backend.modules.memory.domain.enums import MemoryOperation
from backend.modules.memory.infrastructure.models import MemoryAuditLog, MemoryRegistry


class MemoryAuditRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def log_operation(
        self,
        *,
        user_id: str | None,
        agent_id: str | None,
        run_id: str | None,
        project_id: str | None,
        operation: MemoryOperation,
        external_memory_id: str | None,
        memory_level: str | None,
        memory_type: str | None,
        metadata: dict | None,
        source_ref: str | None,
    ) -> MemoryAuditLog:
        entry = MemoryAuditLog(
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            project_id=project_id,
            operation=operation.value,
            external_memory_id=external_memory_id,
            memory_level=memory_level,
            memory_type=memory_type,
            metadata_json=json.dumps(metadata or {}, ensure_ascii=True),
            source_ref=source_ref,
        )
        self.db.add(entry)
        await self.db.flush()
        return entry

    async def list_for_user(
        self,
        user_id: str,
        *,
        limit: int = DEFAULT_PAGE_LIMIT,
        offset: int = 0,
    ) -> tuple[list[MemoryAuditLog], int]:
        stmt = (
            select(MemoryAuditLog)
            .where(MemoryAuditLog.user_id == user_id)
            .order_by(MemoryAuditLog.created_at.desc())
        )
        return await paginate_scalars(self.db, stmt, limit=limit, offset=offset)


class MemoryRegistryRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def register(
        self,
        *,
        external_memory_id: str,
        user_id: str | None,
        agent_id: str | None,
        run_id: str | None,
        project_id: str | None,
        memory_level: str,
        memory_type: str | None,
    ) -> MemoryRegistry:
        existing = await self.get_by_external_id(external_memory_id)
        if existing:
            existing.user_id = user_id
            existing.agent_id = agent_id
            existing.run_id = run_id
            existing.project_id = project_id
            existing.memory_level = memory_level
            existing.memory_type = memory_type
            existing.status = "active"
            existing.updated_at = datetime.now(UTC)
            existing.deleted_at = None
            await self.db.flush()
            return existing

        row = MemoryRegistry(
            external_memory_id=external_memory_id,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            project_id=project_id,
            memory_level=memory_level,
            memory_type=memory_type,
            status="active",
        )
        self.db.add(row)
        await self.db.flush()
        return row

    async def get_by_external_id(self, external_memory_id: str) -> MemoryRegistry | None:
        result = await self.db.execute(
            select(MemoryRegistry).where(
                MemoryRegistry.external_memory_id == external_memory_id,
            )
        )
        return result.scalar_one_or_none()

    async def mark_deleted(self, external_memory_id: str) -> MemoryRegistry | None:
        row = await self.get_by_external_id(external_memory_id)
        if not row:
            return None
        row.status = "deleted"
        row.deleted_at = datetime.now(UTC)
        row.updated_at = datetime.now(UTC)
        await self.db.flush()
        return row

    async def list_for_user(
        self,
        user_id: str,
        *,
        memory_level: str | None = None,
        project_id: str | None = None,
    ) -> list[MemoryRegistry]:
        stmt = select(MemoryRegistry).where(
            MemoryRegistry.user_id == user_id,
            MemoryRegistry.status == "active",
        )
        if memory_level:
            stmt = stmt.where(MemoryRegistry.memory_level == memory_level)
        if project_id:
            stmt = stmt.where(MemoryRegistry.project_id == project_id)
        stmt = stmt.order_by(MemoryRegistry.created_at.desc())
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def user_owns_memory(self, user_id: str, external_memory_id: str) -> bool:
        row = await self.get_by_external_id(external_memory_id)
        if not row or row.status != "active":
            return False
        if row.memory_level == "agent" and row.user_id is None:
            return False
        return row.user_id == user_id
