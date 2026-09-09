"""Phase 8: large-corpus repository / prediction batching helpers."""

from __future__ import annotations

import inspect
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.modules.text_research.application import prediction_service
from backend.modules.text_research.infrastructure.repositories import ResearchRepository


class LargeCorpusRepositoryContractTests(unittest.TestCase):
    def test_list_text_units_by_ids_chunks_in_clause(self) -> None:
        source = inspect.getsource(ResearchRepository.list_text_units_by_ids)
        self.assertIn("iter_item_batches", source)
        self.assertIn("_IN_CLAUSE_BATCH", source)

    def test_list_text_units_for_corpus_pages_large_sets(self) -> None:
        source = inspect.getsource(ResearchRepository.list_text_units_for_corpus)
        self.assertIn("should_use_out_of_core", source)
        self.assertIn("iter_text_units_for_corpus", source)

    def test_annotated_unit_ids_helper_exists(self) -> None:
        self.assertTrue(hasattr(ResearchRepository, "list_annotated_text_unit_ids"))
        source = inspect.getsource(ResearchRepository.list_annotated_text_unit_ids)
        self.assertIn("distinct", source.lower())
        self.assertNotIn("select(Annotation)", source)

    def test_prediction_service_uses_projected_annotation_ids(self) -> None:
        source = inspect.getsource(prediction_service.PredictionService.execute_prediction)
        self.assertIn("list_annotated_text_unit_ids", source)
        self.assertNotIn("list_annotations_for_corpus", source)
        self.assertIn("iter_item_batches", source)


class PredictionRowHelperTests(unittest.TestCase):
    def test_binary_prediction_row_encodes_positive_score(self) -> None:
        row = prediction_service._prediction_row(
            model_id="m1",
            unit_id="u1",
            task_type="binary",
            label_names=["neg", "pos"],
            prediction={"prediction": 1, "probability": 0.9, "uncertainty": 0.1},
        )
        self.assertEqual(row["trained_model_id"], "m1")
        self.assertEqual(row["text_unit_id"], "u1")
        self.assertIn("pos", str(row["scores_json"]))
        self.assertIn("pos", str(row["predicted_labels_json"]))


class IterTextUnitsTests(unittest.IsolatedAsyncioTestCase):
    async def test_iter_stops_on_short_page(self) -> None:
        repo = ResearchRepository(db=MagicMock())
        repo.count_text_units_for_corpus = AsyncMock(return_value=3)
        pages = [
            [MagicMock(id="a"), MagicMock(id="b")],
            [MagicMock(id="c")],
        ]
        repo.list_text_units_for_corpus = AsyncMock(side_effect=pages)

        collected: list[str] = []
        with patch(
            "backend.modules.text_research.infrastructure.repositories.resolve_batch_size",
            return_value=2,
        ):
            async for page in repo.iter_text_units_for_corpus("corpus-1", unit_type="paragraph"):
                collected.extend(unit.id for unit in page)

        self.assertEqual(collected, ["a", "b", "c"])
        self.assertEqual(repo.list_text_units_for_corpus.await_count, 2)


if __name__ == "__main__":
    unittest.main()
