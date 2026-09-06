from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from backend.modules.ai.dependencies import enforce_ai_generation_rate_limit
from backend.modules.ai.evaluation_service import AiEvaluationService
from backend.modules.memory.infrastructure.memory_config import MemoryConfig, validate_memory_config
from backend.modules.rag.application.prompt_context_service import PromptContextService
from backend.modules.rag.application.rag_context_builder import RagContextBuilder
from backend.modules.rag.application.retrieval_ranker import HybridRetrievalRanker
from backend.modules.rag.domain.models import RetrievedChunk
from backend.modules.settings.env_file_store import EnvFileStore


def _chunk(chunk_id: str, content: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        content=content,
        score=score,
        filename="doc.md",
        chunk_index=0,
    )


def test_hybrid_ranker_can_promote_lexically_relevant_candidate() -> None:
    chunks = [
        _chunk("vector", "unrelated material", 0.9),
        _chunk("hybrid", "postgres backup policy", 0.8),
    ]

    ranked = HybridRetrievalRanker().rerank("postgres backup", chunks, limit=1)

    assert ranked[0].chunk_id == "hybrid"


def test_prompt_context_surfaces_both_dependency_failures() -> None:
    service = PromptContextService.__new__(PromptContextService)
    service.rag_config = SimpleNamespace(enabled=True, max_context_tokens=1000)
    service.memory_config = SimpleNamespace(enabled=True)
    service.retrieval = SimpleNamespace(retrieve=AsyncMock(side_effect=RuntimeError("vector down")))
    service.memory = SimpleNamespace(
        recall_for_prompt=AsyncMock(side_effect=RuntimeError("mem down"))
    )
    service.context_builder = RagContextBuilder()

    outcome = asyncio.run(
        service.build(
            query="question",
            user_id="user-1",
            agent_id="default",
            run_id="run-1",
            project_id=None,
            document_ids=None,
            top_k=4,
        )
    )

    assert outcome.retrieval_degraded is True
    assert outcome.memory_degraded is True
    assert outcome.degradation_reason == "retrieval_failed"
    assert outcome.chunks == []


def test_rag_evaluation_combines_output_and_expected_chunk_recall() -> None:
    service = AiEvaluationService.__new__(AiEvaluationService)
    case = SimpleNamespace(
        expected_output_text="PostgreSQL",
        expected_output_json=None,
        expected_chunk_ids_json=["chunk-1", "chunk-2"],
    )

    score, passed, note = service._score_evaluation_case("PostgreSQL", None, ["chunk-1"], case)

    assert score == 0.85
    assert passed is False
    assert "0.50" in note


def test_rag_evaluation_supports_retrieval_only_cases() -> None:
    service = AiEvaluationService.__new__(AiEvaluationService)
    case = SimpleNamespace(
        expected_output_text=None,
        expected_output_json=None,
        expected_chunk_ids_json=["chunk-1"],
    )

    score, passed, _ = service._score_evaluation_case("anything", None, ["chunk-1"], case)

    assert score == 1.0
    assert passed is True


def test_memory_config_validation_rejects_invalid_bounds() -> None:
    base = dict(
        enabled=True,
        write_enabled=True,
        audit_enabled=True,
        default_limit=5,
        min_confidence=0.5,
        session_ttl_days=30,
        mem0_mode="oss",
        mem0_api_key="",
        mem0_org_id="",
        mem0_project_id="",
        mem0_base_url="",
        app_id="test",
    )
    with pytest.raises(RuntimeError, match="between 0 and 1"):
        validate_memory_config(MemoryConfig(**{**base, "min_confidence": 1.1}))
    with pytest.raises(RuntimeError, match="Unsupported MEM0_MODE"):
        validate_memory_config(MemoryConfig(**{**base, "mem0_mode": "invalid"}))


def test_env_file_store_preserves_comments_and_updates_atomically(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("# heading\nA=old\nKEEP=value\n", encoding="utf-8")

    def parse(line: str):
        if "=" not in line or line.lstrip().startswith("#"):
            return None
        key, value = line.split("=", 1)
        return key, value

    EnvFileStore(env_path).update({"A": "new", "B": "added"}, parse_line=parse)

    assert env_path.read_text(encoding="utf-8") == ("# heading\nA=new\nKEEP=value\n\nB=added\n")


def test_generation_rate_limit_is_scoped_by_user_and_ip() -> None:
    request = SimpleNamespace(client=SimpleNamespace(host="203.0.113.8"))
    user = SimpleNamespace(id="user-1")
    with (
        patch("backend.modules.ai.dependencies.settings.AI_RATE_LIMIT_REQUESTS", 12),
        patch("backend.modules.ai.dependencies.settings.AI_RATE_LIMIT_WINDOW_SECONDS", 60),
        patch("backend.modules.ai.dependencies.check_rate_limit", AsyncMock()) as limiter,
    ):
        asyncio.run(enforce_ai_generation_rate_limit(request, user))

    limiter.assert_awaited_once_with("rate_limit:ai:user-1:203.0.113.8", 12, 60)
