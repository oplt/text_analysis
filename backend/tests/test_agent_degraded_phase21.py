"""Phase 21: agent run persists degraded retrieval/memory flags."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from backend.modules.ai.application.agent_service import AgentService
from backend.modules.ai.application.prompt_context_builder import AgentPromptContext


class AgentDegradedRunTests(unittest.IsolatedAsyncioTestCase):
    @patch("backend.modules.ai.application.agent_service.queue_turn_memory_extraction")
    @patch("backend.modules.ai.application.agent_service.MemoryConfig.from_settings")
    @patch("backend.modules.ai.application.agent_service.AiService")
    @patch("backend.modules.ai.application.agent_service.MemoryService")
    async def test_run_records_retrieval_and_memory_degraded(
        self,
        memory_cls: MagicMock,
        ai_cls: MagicMock,
        config_fn: MagicMock,
        queue_fn: MagicMock,
    ) -> None:
        config_fn.return_value = SimpleNamespace(enabled=True, write_enabled=False)
        ai_instance = ai_cls.return_value
        ai_instance.run_prompt = AsyncMock(
            return_value=SimpleNamespace(
                id="run-1",
                output_text="answer",
                retrieval_degraded=True,
                memory_degraded=True,
            )
        )

        service = AgentService(db=AsyncMock())
        service.prompt_context = MagicMock()
        service.prompt_context.build = AsyncMock(
            return_value=AgentPromptContext(
                additional_system_context="",
                retrieved_chunk_ids=[],
                retrieved_memories=[],
                effective_retrieval_query="q",
                retrieval_degraded=True,
                memory_degraded=True,
                degradation_reason="vector_unavailable",
                injection_chunks_filtered=0,
            )
        )

        user = SimpleNamespace(id="user-a")
        run, run_id, _working = await service.run_agent_prompt(
            user,
            prompt_template_key="assistant",
            prompt_version_id=None,
            variables={},
            retrieval_query="q",
            document_ids=[],
            top_k=4,
            review_required=False,
        )

        self.assertEqual(run.id, "run-1")
        self.assertTrue(run_id)
        call_kwargs = ai_instance.run_prompt.await_args.kwargs
        self.assertTrue(call_kwargs["retrieval_degraded"])
        self.assertTrue(call_kwargs["memory_degraded"])
        queue_fn.assert_not_called()


if __name__ == "__main__":
    unittest.main()
