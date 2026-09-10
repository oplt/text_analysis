import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from backend.modules.ai.application.agent_service import AgentService
from backend.modules.ai.application.prompt_context_builder import (
    AgentPromptContext,
    AgentPromptContextBuilder,
)
from backend.modules.rag.application.prompt_context_service import PromptContextOutcome
from backend.modules.rag.domain.models import RetrievedChunk


class AgentPromptContextBuilderTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.db = AsyncMock()
        self.memory = MagicMock()
        self.memory.recall_for_prompt = AsyncMock(return_value=("memory block", [], False))
        self.builder = AgentPromptContextBuilder(self.db, self.memory)

    @staticmethod
    def _chunk() -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            content="document block",
            score=0.9,
            filename="doc.txt",
            chunk_index=0,
        )

    @patch(
        "backend.modules.ai.application.prompt_context_builder.RagConfig.from_settings",
        return_value=SimpleNamespace(enabled=True),
    )
    async def test_single_memory_recall_and_assembled_context(self, _rag_config):
        self.builder.context.build = AsyncMock(
            return_value=PromptContextOutcome(
                system_context="memory block\n\ndocument block",
                document_context="document block",
                memory_context="memory block",
                chunks=[self._chunk()],
            )
        )
        context = await self.builder.build(
            user_id="user-1",
            agent_id="default",
            query="hello",
            run_id="run-1",
            project_id="proj-1",
            retrieval_query="hello",
            document_ids=["doc-1"],
            top_k=3,
        )

        self.builder.context.build.assert_awaited_once()
        self.assertIsNone(context.effective_retrieval_query)
        self.assertIn("memory block", context.additional_system_context or "")
        self.assertIn("document block", context.additional_system_context or "")

    @patch(
        "backend.modules.ai.application.prompt_context_builder.RagConfig.from_settings",
        return_value=SimpleNamespace(enabled=False),
    )
    async def test_legacy_retrieval_when_rag_disabled(self, _rag_config):
        self.builder.context.build = AsyncMock(
            return_value=PromptContextOutcome(
                system_context=None,
                document_context="",
                memory_context="",
            )
        )
        context = await self.builder.build(
            user_id="user-1",
            agent_id="default",
            query="hello",
            run_id="run-1",
            project_id=None,
            retrieval_query="search",
            document_ids=[],
            top_k=4,
        )
        self.assertEqual(context.effective_retrieval_query, "search")


class AgentRetrievalConsolidationTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.db = AsyncMock()
        self.user = MagicMock(id="user-1")

    @patch("backend.modules.ai.application.agent_service.AgentPromptContextBuilder")
    async def test_uses_unified_prompt_context(self, builder_cls):
        builder = builder_cls.return_value
        builder.build = AsyncMock(
            return_value=AgentPromptContext(
                additional_system_context="combined context",
                effective_retrieval_query=None,
                retrieved_memories=[],
            )
        )

        service = AgentService(self.db)
        service.config = SimpleNamespace(enabled=True, write_enabled=False)
        service.ai = MagicMock()
        service.ai.run_prompt = AsyncMock(return_value=MagicMock(id="run-1", output_text="ok"))

        await service.run_agent_prompt(
            self.user,
            prompt_template_key="demo",
            prompt_version_id=None,
            variables={"user_message": "hello"},
            retrieval_query="search me",
            document_ids=["doc-1"],
            top_k=3,
            review_required=False,
            agent_id="research-agent",
            run_id="memory-run-1",
        )

        builder.build.assert_awaited_once()
        build_kwargs = builder.build.await_args.kwargs
        self.assertEqual(build_kwargs["document_ids"], ["doc-1"])
        self.assertEqual(build_kwargs["top_k"], 3)

        run_kwargs = service.ai.run_prompt.await_args.kwargs
        self.assertIsNone(run_kwargs["retrieval_query"])
        self.assertEqual(run_kwargs["additional_system_context"], "combined context")
        self.assertEqual(run_kwargs["agent_id"], "research-agent")
        self.assertEqual(run_kwargs["agent_run_id"], "memory-run-1")

    @patch("backend.modules.ai.application.agent_service.AgentPromptContextBuilder")
    async def test_agent_reuses_prompt_context_memories_without_second_recall(
        self,
        builder_cls,
    ):
        memory_item = MagicMock()
        builder_cls.return_value.build = AsyncMock(
            return_value=AgentPromptContext(
                additional_system_context="memory ctx",
                effective_retrieval_query=None,
                retrieved_memories=[memory_item],
            )
        )
        service = AgentService(self.db)
        service.config = SimpleNamespace(enabled=True, write_enabled=False)
        service.ai = MagicMock()
        service.ai.run_prompt = AsyncMock(return_value=MagicMock(id="run-1", output_text="ok"))

        _run, _run_id, working = await service.run_agent_prompt(
            self.user,
            prompt_template_key="demo",
            prompt_version_id=None,
            variables={"user_message": "hello"},
            retrieval_query="search me",
            document_ids=["doc-1"],
            top_k=3,
            review_required=False,
        )

        self.assertEqual(working.retrieved_memories, [memory_item])

    @patch("backend.modules.ai.application.agent_service.AgentPromptContextBuilder")
    async def test_working_memory_populated_from_prompt_context(self, builder_cls):
        memory_item = MagicMock()
        builder = builder_cls.return_value
        builder.build = AsyncMock(
            return_value=AgentPromptContext(
                additional_system_context="combined context",
                effective_retrieval_query=None,
                retrieved_memories=[memory_item],
            )
        )

        service = AgentService(self.db)
        service.config = SimpleNamespace(enabled=True, write_enabled=False)
        service.ai = MagicMock()
        service.ai.run_prompt = AsyncMock(return_value=MagicMock(id="run-1", output_text="ok"))
        service.memory.recall = AsyncMock()

        _run, _run_id, working = await service.run_agent_prompt(
            self.user,
            prompt_template_key="demo",
            prompt_version_id=None,
            variables={"user_message": "hello"},
            retrieval_query=None,
            document_ids=[],
            top_k=3,
            review_required=False,
        )

        service.memory.recall.assert_not_called()
        self.assertEqual(working.retrieved_memories, [memory_item])
