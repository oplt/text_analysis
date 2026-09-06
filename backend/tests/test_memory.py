import asyncio
import unittest
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from backend.modules.memory.application.memory_consolidator import MemoryConsolidator
from backend.modules.memory.application.memory_context_builder import MemoryContextBuilder
from backend.modules.memory.application.memory_extractor import MemoryExtractor
from backend.modules.memory.application.memory_router import MemoryRouter
from backend.modules.memory.application.memory_service import MemoryService
from backend.modules.memory.domain.enums import MemoryLevel, MemoryPrivacy, MemoryType
from backend.modules.memory.domain.models import MemoryItem, MemoryMetadata, MemorySearchRequest
from backend.modules.memory.domain.policies import contains_secret, evaluate_storage_policy
from backend.modules.memory.infrastructure.mem0_client import (
    MEMORY_CONTEXT_HEADER,
    NullMem0Adapter,
    build_entity_filters,
)
from backend.modules.memory.infrastructure.memory_config import MemoryConfig


def _item(
    memory_id: str,
    content: str,
    *,
    user_id: str = "user-a",
    level: MemoryLevel = MemoryLevel.USER,
    run_id: str | None = None,
    project_id: str | None = None,
    score: float = 0.9,
) -> MemoryItem:
    return MemoryItem(
        id=memory_id,
        content=content,
        score=score,
        metadata=MemoryMetadata(
            memory_level=level,
            memory_type=MemoryType.FACT,
            user_id=user_id,
            agent_id="default",
            run_id=run_id,
            project_id=project_id,
            confidence=0.9,
            created_at=datetime.now(UTC),
        ),
    )


class MemoryPolicyTest(unittest.TestCase):
    def test_secrets_rejected(self):
        self.assertTrue(contains_secret("my api_key=sk-abcdefghijklmnopqrstuvwxyz"))
        decision = evaluate_storage_policy(
            content="password=supersecret123",
            memory_level=MemoryLevel.USER,
            memory_type=MemoryType.FACT,
            confidence=0.9,
            privacy=MemoryPrivacy.NORMAL,
            min_confidence=0.65,
        )
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "secret_detected")


class MemoryRouterTest(unittest.TestCase):
    def test_preference_routes_to_user_memory(self):
        routed = MemoryRouter().route(content="User prefers beginner-friendly explanations.")
        self.assertEqual(routed.memory_level, MemoryLevel.USER)
        self.assertEqual(routed.memory_type, MemoryType.PREFERENCE)

    def test_task_state_routes_to_session_memory(self):
        routed = MemoryRouter().route(content="Current plan: finish auth middleware next step.")
        self.assertEqual(routed.memory_level, MemoryLevel.SESSION)
        self.assertEqual(routed.memory_type, MemoryType.TASK_STATE)

    def test_project_decision_routes_to_project_memory(self):
        routed = MemoryRouter().route(
            content="For project X, the chosen DB is PostgreSQL.",
            is_project_decision=True,
        )
        self.assertEqual(routed.memory_level, MemoryLevel.PROJECT)
        self.assertEqual(routed.memory_type, MemoryType.DECISION)


class MemoryConsolidatorTest(unittest.TestCase):
    def test_duplicate_memories_consolidated(self):
        items = [
            _item("1", "User prefers FastAPI examples."),
            _item("2", "user prefers fastapi examples."),
        ]
        consolidated = MemoryConsolidator().consolidate(items)
        self.assertEqual(len(consolidated), 1)


class MemoryContextBuilderTest(unittest.IsolatedAsyncioTestCase):
    async def test_memory_retrieval_included_in_prompt_context(self):
        mock_service = AsyncMock()
        mock_service.recall = AsyncMock(return_value=[_item("u1", "User likes concise answers.")])
        builder = MemoryContextBuilder(mock_service)
        block = await builder.build_context_block(
            MemorySearchRequest(
                user_id="user-a",
                agent_id="default",
                query="FastAPI help",
            )
        )
        mock_service.recall.assert_awaited_once()
        self.assertIn(MEMORY_CONTEXT_HEADER, block)
        self.assertIn("[memory:u1]", block)
        self.assertIn("untrusted background material", block)

    async def test_recall_ranked_uses_single_recall_with_all_levels(self):
        mock_service = AsyncMock()
        mock_service.recall = AsyncMock(return_value=[])
        builder = MemoryContextBuilder(mock_service)
        await builder.recall_ranked(
            MemorySearchRequest(
                user_id="user-a",
                agent_id="default",
                query="help",
            )
        )
        mock_service.recall.assert_awaited_once()
        levels = mock_service.recall.await_args.kwargs["memory_levels"]
        self.assertEqual(
            levels,
            ["user", "project", "session", "agent", "episodic"],
        )


class MemoryRecallForPromptTest(unittest.IsolatedAsyncioTestCase):
    @patch("backend.modules.memory.application.memory_service.Mem0Client")
    async def test_recall_for_prompt_returns_context_and_items_once(self, _mem0_cls):
        config = MemoryConfig(
            enabled=True,
            write_enabled=True,
            audit_enabled=False,
            default_limit=10,
            min_confidence=0.65,
            session_ttl_days=30,
            mem0_mode="oss",
            mem0_api_key="",
            mem0_org_id="",
            mem0_project_id="",
            mem0_base_url="",
            app_id="test",
        )
        service = MemoryService(db=AsyncMock(), config=config)
        expected_items = [_item("m1", "Prefers concise answers.")]
        service.context_builder.recall_ranked = AsyncMock(return_value=expected_items)
        service.context_builder.format_context_block = MagicMock(return_value="formatted block")

        context, items, degraded = await service.recall_for_prompt(
            MemorySearchRequest(
                user_id="user-a",
                agent_id="default",
                query="help",
            )
        )

        service.context_builder.recall_ranked.assert_awaited_once()
        service.context_builder.format_context_block.assert_called_once_with(expected_items)
        self.assertEqual(context, "formatted block")
        self.assertEqual(items, expected_items)
        self.assertFalse(degraded)


class Mem0FilterTest(unittest.TestCase):
    def test_session_memory_requires_run_id_filter(self):
        filters = build_entity_filters(
            memory_level=MemoryLevel.SESSION,
            user_id="user-a",
            agent_id="default",
            run_id="run-1",
            project_id=None,
            app_id="app",
        )
        self.assertEqual(filters["run_id"], "run-1")
        self.assertEqual(filters["memory_level"], "session")

    def test_project_memory_includes_project_id(self):
        filters = build_entity_filters(
            memory_level=MemoryLevel.PROJECT,
            user_id="user-a",
            agent_id="default",
            run_id=None,
            project_id="proj-1",
            app_id="app",
        )
        self.assertEqual(filters["project_id"], "proj-1")


class MemoryServiceIsolationTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.db = AsyncMock()
        self.config = SimpleNamespace(
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

    def _service_with_items(self, items_by_level: dict[MemoryLevel, list[MemoryItem]]):
        service = MemoryService(self.db, config=self.config)
        service.mem0 = MagicMock()
        service.mem0.available = True

        async def search(**kwargs):
            return items_by_level.get(kwargs["memory_level"], [])

        service.mem0.search = AsyncMock(side_effect=search)
        service.mem0.get = AsyncMock(return_value=None)
        service.mem0.delete = AsyncMock()
        service.mem0.add = AsyncMock(return_value={"id": "new-mem"})
        service.project_access = MagicMock()
        service.project_access.get_project_for_user = AsyncMock(
            return_value=SimpleNamespace(id="proj-1")
        )
        service.project_access.filter_accessible_project_ids = AsyncMock(
            side_effect=lambda _user_id, project_ids: set(project_ids)
        )
        service.audit_repo = MagicMock()
        service.registry_repo = MagicMock()
        service.registry_repo.register = AsyncMock()
        service.registry_repo.get_by_external_id = AsyncMock(return_value=None)
        service.registry_repo.mark_deleted = AsyncMock()
        service.db.commit = AsyncMock()
        return service

    async def test_user_a_cannot_read_user_b_memory(self):
        service = self._service_with_items(
            {
                MemoryLevel.USER: [
                    _item("b1", "User B secret pref", user_id="user-b"),
                ]
            }
        )
        items = await service.recall(
            user_id="user-a",
            agent_id="default",
            query="preferences",
            memory_levels=["user"],
        )
        self.assertEqual(items, [])

    async def test_session_memory_only_for_matching_run_id(self):
        service = self._service_with_items(
            {
                MemoryLevel.SESSION: [
                    _item("s1", "Current task", run_id="run-1"),
                ]
            }
        )
        items = await service.recall(
            user_id="user-a",
            agent_id="default",
            query="task",
            run_id="run-2",
            memory_levels=["session"],
        )
        self.assertEqual(len(items), 1)
        service.mem0.search.assert_called()
        call_kwargs = service.mem0.search.await_args.kwargs
        self.assertEqual(call_kwargs["run_id"], "run-2")

    async def test_project_memory_requires_project_id(self):
        service = self._service_with_items(
            {
                MemoryLevel.PROJECT: [
                    _item("p1", "Uses PostgreSQL", project_id="proj-1"),
                ]
            }
        )
        items = await service.recall(
            user_id="user-a",
            agent_id="default",
            query="database",
            project_id="proj-1",
            memory_levels=["project"],
        )
        self.assertEqual(len(items), 1)
        service.project_access.get_project_for_user.assert_called_with("proj-1", "user-a")

    async def test_recall_searches_multiple_levels_in_parallel(self):
        service = self._service_with_items(
            {
                MemoryLevel.USER: [_item("u1", "Prefers concise answers.")],
                MemoryLevel.AGENT: [_item("a1", "Use bullet lists.")],
            }
        )
        with patch("asyncio.gather", wraps=asyncio.gather) as gather_fn:
            items = await service.recall(
                user_id="user-a",
                agent_id="default",
                query="help",
                memory_levels=["user", "agent"],
            )
        gather_fn.assert_called()
        self.assertEqual(len(items), 2)
        self.assertEqual(service.mem0.search.await_count, 2)

    async def test_recall_bulk_authorizes_project_memories_with_one_query(self):
        service = self._service_with_items(
            {
                MemoryLevel.PROJECT: [
                    _item(
                        "p1",
                        "Uses PostgreSQL",
                        level=MemoryLevel.PROJECT,
                        project_id="proj-1",
                    ),
                    _item(
                        "p2",
                        "Uses Redis",
                        level=MemoryLevel.PROJECT,
                        project_id="proj-2",
                    ),
                ]
            }
        )
        service.project_access.filter_accessible_project_ids = AsyncMock(return_value={"proj-1"})

        items = await service.recall(
            user_id="user-a",
            agent_id="default",
            query="database",
            project_id="proj-1",
            memory_levels=["project"],
        )

        service.project_access.filter_accessible_project_ids.assert_awaited_once()
        requested_ids = service.project_access.filter_accessible_project_ids.await_args.args[1]
        self.assertEqual(requested_ids, {"proj-1", "proj-2"})
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].id, "p1")

    async def test_deleted_memory_not_returned_from_registry_get(self):
        service = self._service_with_items({})
        service.registry_repo.get_by_external_id = AsyncMock(
            return_value=SimpleNamespace(
                status="deleted",
                user_id="user-a",
                project_id=None,
                agent_id="default",
                run_id=None,
                memory_level="user",
                memory_type="fact",
            )
        )
        from fastapi import HTTPException

        with self.assertRaises(HTTPException) as ctx:
            await service.get_memory(user_id="user-a", memory_id="gone-1")
        self.assertEqual(ctx.exception.status_code, 404)

    async def test_secrets_not_stored(self):
        service = self._service_with_items({})
        result = await service.remember(
            user_id="user-a",
            agent_id="default",
            content="api_key=sk-abcdefghijklmnopqrstuvwxyz123456",
            memory_level="user",
        )
        self.assertFalse(result.accepted)
        service.mem0.add.assert_not_called()


class MemoryExtractorTest(unittest.TestCase):
    def test_extracts_durable_preference_not_every_message(self):
        extractor = MemoryExtractor()
        candidates = extractor.extract_from_turn(
            user_message="Hello!",
            assistant_message="User prefers beginner-friendly FastAPI examples.",
        )
        self.assertTrue(any("prefers" in c.content.lower() for c in candidates))
        self.assertFalse(any("hello" in c.content.lower() for c in candidates))


class NullMem0AdapterTest(unittest.IsolatedAsyncioTestCase):
    async def test_ai_works_when_mem0_unavailable(self):
        adapter = NullMem0Adapter("not configured")
        self.assertFalse(adapter.available)
        results = await adapter.search(query="test", filters={"user_id": "a"}, top_k=5)
        self.assertEqual(results, [])


class AgentServiceDegradedTest(unittest.IsolatedAsyncioTestCase):
    @patch("backend.modules.rag.application.prompt_context_service.MemoryConfig.from_settings")
    @patch("backend.modules.ai.application.agent_service.MemoryConfig.from_settings")
    @patch("backend.modules.ai.application.agent_service.AiService")
    @patch("backend.modules.ai.application.agent_service.MemoryService")
    async def test_agent_run_without_memory(
        self, memory_cls, ai_cls, agent_config_fn, builder_config_fn
    ):
        disabled = SimpleNamespace(enabled=False, write_enabled=False)
        agent_config_fn.return_value = disabled
        builder_config_fn.return_value = disabled
        ai_instance = ai_cls.return_value
        ai_instance.run_prompt = AsyncMock(
            return_value=SimpleNamespace(id="run-1", output_text="ok")
        )
        memory_instance = memory_cls.return_value
        memory_instance.recall_for_prompt = AsyncMock()

        from backend.modules.ai.application.agent_service import AgentService

        service = AgentService(db=AsyncMock())
        user = SimpleNamespace(id="user-a")
        run, run_id, _ = await service.run_agent_prompt(
            user,
            prompt_template_key="assistant",
            prompt_version_id=None,
            variables={},
            retrieval_query=None,
            document_ids=[],
            top_k=4,
            review_required=False,
        )
        self.assertEqual(run.output_text, "ok")
        memory_instance.recall_for_prompt.assert_not_awaited()
        memory_instance.build_prompt_context.assert_not_called()
        ai_instance.run_prompt.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
