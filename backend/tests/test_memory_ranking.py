import unittest
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

from backend.lib.memory_ranking import memory_recency_weight
from backend.modules.memory.application.memory_context_builder import MemoryContextBuilder
from backend.modules.memory.domain.enums import MemoryLevel, MemoryType
from backend.modules.memory.domain.models import MemoryItem, MemoryMetadata, MemorySearchRequest
from backend.modules.rag.application.rag_context_builder import RagContextBuilder
from backend.modules.rag.domain.models import RetrievedChunk


class MemoryRecencyWeightTest(unittest.TestCase):
    def test_fresh_memory_scores_higher_than_old(self):
        now = datetime.now(UTC)
        fresh = memory_recency_weight(now)
        stale = memory_recency_weight(now - timedelta(days=90))
        self.assertGreater(fresh, stale)
        self.assertAlmostEqual(fresh, 1.0, places=2)

    def test_missing_timestamp_returns_zero(self):
        self.assertEqual(memory_recency_weight(None), 0.0)


class MemoryRankScoreTest(unittest.IsolatedAsyncioTestCase):
    async def test_recent_memory_ranks_above_stale_with_same_level(self):
        now = datetime.now(UTC)
        recent = MemoryItem(
            id="recent",
            content="Recent fact",
            score=0.8,
            metadata=MemoryMetadata(
                memory_level=MemoryLevel.USER,
                memory_type=MemoryType.FACT,
                user_id="user-a",
                agent_id="default",
                created_at=now,
            ),
        )
        stale = MemoryItem(
            id="stale",
            content="Old fact",
            score=0.8,
            metadata=MemoryMetadata(
                memory_level=MemoryLevel.USER,
                memory_type=MemoryType.FACT,
                user_id="user-a",
                agent_id="default",
                created_at=now - timedelta(days=120),
            ),
        )
        mock_service = AsyncMock()
        mock_service.recall = AsyncMock(return_value=[stale, recent])
        builder = MemoryContextBuilder(mock_service)
        ranked = await builder.recall_ranked(
            MemorySearchRequest(user_id="user-a", agent_id="default", query="fact")
        )
        self.assertEqual(ranked[0].id, "recent")


class RagContextBudgetTest(unittest.TestCase):
    def _chunk(self, chunk_id: str, content: str, score: float) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=chunk_id,
            document_id="doc-1",
            content=content,
            score=score,
            filename="notes.txt",
            chunk_index=0,
        )

    def test_trim_chunks_respects_token_budget(self):
        builder = RagContextBuilder()
        chunks = [
            self._chunk("c1", "A" * 4000, 0.9),
            self._chunk("c2", "B" * 4000, 0.8),
            self._chunk("c3", "C" * 4000, 0.7),
        ]
        trimmed = builder.trim_chunks_to_token_budget(chunks, max_tokens=1200, reserved_tokens=200)
        self.assertGreaterEqual(len(trimmed), 1)
        self.assertLess(len(trimmed), len(chunks))

    def test_trim_chunks_prefers_higher_scores(self):
        builder = RagContextBuilder()
        chunks = [
            self._chunk("low", "x" * 800, 0.2),
            self._chunk("high", "y" * 800, 0.95),
        ]
        trimmed = builder.trim_chunks_to_token_budget(chunks, max_tokens=600, reserved_tokens=100)
        self.assertEqual(trimmed[0].chunk_id, "high")
