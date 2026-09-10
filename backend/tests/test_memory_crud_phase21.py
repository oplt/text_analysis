"""Phase 21: memory create / list / search / delete / isolation coverage."""

from __future__ import annotations

import unittest
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from fastapi import HTTPException

from backend.modules.memory.application.memory_service import MemoryService
from backend.modules.memory.domain.enums import MemoryLevel, MemoryType
from backend.modules.memory.domain.models import MemoryItem, MemoryMetadata


def _item(memory_id: str, content: str, *, user_id: str = "user-a") -> MemoryItem:
    return MemoryItem(
        id=memory_id,
        content=content,
        score=0.9,
        metadata=MemoryMetadata(
            memory_level=MemoryLevel.USER,
            memory_type=MemoryType.FACT,
            user_id=user_id,
            agent_id="default",
            run_id=None,
            project_id=None,
            confidence=0.9,
            created_at=datetime.now(UTC),
        ),
    )


class MemoryCrudIsolationTests(unittest.IsolatedAsyncioTestCase):
    def _service(self) -> MemoryService:
        config = SimpleNamespace(
            enabled=True,
            write_enabled=True,
            audit_enabled=False,
            default_limit=10,
            min_confidence=0.65,
            session_ttl_days=30,
            mem0_configured=True,
            mem0_mode="hosted",
            mem0_api_key="",
            mem0_base_url="",
            app_id="test",
        )
        service = MemoryService(AsyncMock(), config=config)
        service.mem0 = MagicMock()
        service.mem0.available = True
        service.mem0.add = AsyncMock(return_value={"id": "mem-new"})
        service.mem0.search = AsyncMock(return_value=[_item("mem-1", "likes FastAPI")])
        service.mem0.get_all = AsyncMock(return_value=[_item("mem-1", "likes FastAPI")])
        service.mem0.delete = AsyncMock()
        service.mem0.get = AsyncMock(return_value=_item("mem-1", "likes FastAPI"))
        service.project_access = MagicMock()
        service.project_access.get_project_for_user = AsyncMock(
            return_value=SimpleNamespace(id="proj-1")
        )
        service.project_access.filter_accessible_project_ids = AsyncMock(
            side_effect=lambda _uid, ids: set(ids)
        )
        service.audit_repo = MagicMock()
        service.registry_repo = MagicMock()
        service.registry_repo.register = AsyncMock()
        service.registry_repo.get_by_external_id = AsyncMock(
            return_value=SimpleNamespace(
                status="active",
                user_id="user-a",
                project_id=None,
                agent_id="default",
                run_id=None,
                memory_level="user",
                memory_type="fact",
            )
        )
        service.registry_repo.mark_deleted = AsyncMock()
        service.db.commit = AsyncMock()
        return service

    async def test_create_list_search_delete(self) -> None:
        service = self._service()

        created = await service.remember(
            user_id="user-a",
            agent_id="default",
            content="User prefers FastAPI examples.",
            memory_level="user",
        )
        self.assertTrue(created.accepted)
        service.mem0.add.assert_awaited()

        items, total = await service.list_user_memories(user_id="user-a", agent_id="default")
        self.assertGreaterEqual(total, 1)
        self.assertEqual(items[0].id, "mem-1")

        searched = await service.recall(
            user_id="user-a",
            agent_id="default",
            query="FastAPI",
            memory_levels=["user"],
        )
        self.assertEqual(len(searched), 1)

        await service.forget(user_id="user-a", memory_id="mem-1", reason="test")
        service.mem0.delete.assert_awaited_once_with("mem-1")
        service.registry_repo.mark_deleted.assert_awaited()

    async def test_forget_rejects_cross_user(self) -> None:
        service = self._service()
        service.registry_repo.get_by_external_id = AsyncMock(
            return_value=SimpleNamespace(
                status="active",
                user_id="user-b",
                project_id=None,
                agent_id="default",
                run_id=None,
                memory_level="user",
                memory_type="fact",
            )
        )
        with self.assertRaises(HTTPException) as ctx:
            await service.forget(user_id="user-a", memory_id="mem-b", reason="nope")
        self.assertEqual(ctx.exception.status_code, 403)


if __name__ == "__main__":
    unittest.main()
