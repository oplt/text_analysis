import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from backend.modules.ai.application.rag_answer_prompt import (
    DEFAULT_RAG_ANSWER_SYSTEM_PROMPT,
    resolve_rag_answer_prompt,
)
from backend.modules.ai.service import AiService
from backend.modules.identity_access.models import User


class RagAnswerPromptTest(unittest.IsolatedAsyncioTestCase):
    async def test_resolve_rag_answer_prompt_falls_back_to_defaults(self):
        repo = MagicMock()
        repo.get_prompt_template_by_key_for_user = AsyncMock(return_value=None)
        user = User(id="user-1", email="a@example.com", password_hash="x")

        spec = await resolve_rag_answer_prompt(repo, user)

        self.assertIsNone(spec.template_id)
        self.assertEqual(spec.system_prompt, DEFAULT_RAG_ANSWER_SYSTEM_PROMPT)

    async def test_resolve_rag_answer_prompt_selects_active_version_from_paginated_result(self):
        repo = MagicMock()
        repo.get_prompt_template_by_key_for_user = AsyncMock(
            return_value=SimpleNamespace(id="tpl-1", active_version_id="ver-active")
        )
        active = SimpleNamespace(
            id="ver-active",
            provider_key="openai",
            model_name="gpt-test",
            system_prompt="Custom RAG prompt",
            response_format="text",
            temperature=0.1,
            input_cost_per_million=1,
            output_cost_per_million=2,
            is_published=True,
        )
        repo.list_prompt_versions = AsyncMock(return_value=([active], 1))
        user = User(id="user-1", email="a@example.com", password_hash="x")

        with patch(
            "backend.modules.ai.application.rag_answer_prompt.settings.RAG_ASK_PROMPT_TEMPLATE_KEY",
            "rag-answer",
        ):
            spec = await resolve_rag_answer_prompt(repo, user)

        repo.list_prompt_versions.assert_awaited_once_with("tpl-1", limit=200, offset=0)
        self.assertEqual(spec.template_id, "tpl-1")
        self.assertEqual(spec.version_id, "ver-active")
        self.assertEqual(spec.system_prompt, "Custom RAG prompt")


class AiServiceRagAnswerTest(unittest.IsolatedAsyncioTestCase):
    @patch("backend.modules.ai.run_service.resolve_rag_answer_prompt")
    async def test_run_rag_answer_creates_run_with_retrieved_chunk_ids(self, resolve_prompt):
        db = AsyncMock()
        service = AiService(db)
        service.repo = MagicMock()
        service.repo.create_run = AsyncMock(
            return_value=SimpleNamespace(id="run-1", provider_key="local")
        )
        service.providers = MagicMock()
        service.providers.get.return_value.generate = AsyncMock(
            return_value=SimpleNamespace(
                output_text="Answer text",
                output_json=None,
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
            )
        )
        service._finalize_run_generation = AsyncMock(
            return_value=SimpleNamespace(
                id="run-1",
                output_text="Answer text",
                model_name="local-heuristic",
            )
        )
        resolve_prompt.return_value = SimpleNamespace(
            template_id=None,
            version_id=None,
            provider_key="local",
            model_name="local-heuristic",
            system_prompt=DEFAULT_RAG_ANSWER_SYSTEM_PROMPT,
            response_format="text",
            execution_version=SimpleNamespace(
                model_name="local-heuristic",
                response_format="text",
                temperature=0.2,
                input_cost_per_million=0,
                output_cost_per_million=0,
            ),
        )
        user = User(id="user-1", email="a@example.com", password_hash="x")

        run = await service.run_rag_answer(
            user,
            query="What is Postgres?",
            combined_context="User question:\nWhat is Postgres?",
            retrieved_chunk_ids=["chunk-1"],
            retrieval_degraded=True,
            memory_degraded=True,
            degradation_reason="provider_partial",
            injection_chunks_filtered=2,
        )

        self.assertEqual(run.id, "run-1")
        create_kwargs = service.repo.create_run.await_args.kwargs
        self.assertEqual(create_kwargs["retrieved_chunk_ids_json"], ["chunk-1"])
        self.assertEqual(create_kwargs["retrieval_query"], "What is Postgres?")
        self.assertTrue(create_kwargs["retrieval_degraded"])
        self.assertTrue(create_kwargs["memory_degraded"])
        self.assertEqual(create_kwargs["degradation_reason"], "provider_partial")
        self.assertEqual(create_kwargs["injection_chunks_filtered"], 2)
        self.assertEqual(create_kwargs["input_messages_json"][0]["role"], "system")
        self.assertIn("What is Postgres?", create_kwargs["input_messages_json"][1]["content"])
        self.assertNotIn("User question:", create_kwargs["input_messages_json"][1]["content"])
        service._finalize_run_generation.assert_awaited_once()
