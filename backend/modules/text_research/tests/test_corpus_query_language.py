"""Corpus language filter vs KWIC query_language separation (TASK-002)."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from pydantic import ValidationError

from backend.modules.text_research.api.quantitative_routes import _analysis_filters
from backend.modules.text_research.api.schemas import FrequencyRequest, KwicRequest
from backend.modules.text_research.application.quantitative_analysis_service import (
    QuantitativeAnalysisService,
    _apply_document_filters,
    _filter_kwargs,
)


class AnalysisFiltersLanguageTests(unittest.TestCase):
    def test_analysis_filters_preserves_corpus_language(self) -> None:
        body = FrequencyRequest(unit_type="paragraph", language="en", top_n=10)
        filters = _analysis_filters(body)
        self.assertEqual(filters.get("language"), "en")
        self.assertNotIn("top_n", filters)
        self.assertNotIn("unit_type", filters)

    def test_filter_kwargs_maps_language_for_document_select(self) -> None:
        kwargs = _filter_kwargs({"language": "en", "organization": "UNESCO"})
        self.assertEqual(kwargs["language"], "en")
        self.assertEqual(kwargs["organization"], "UNESCO")

    def test_apply_document_filters_selects_english_only(self) -> None:
        docs = [
            SimpleNamespace(id="d1", language="en", organization=None, organization_type=None,
                            region=None, cultural_sphere=None, publication_type=None,
                            country=None, publication_year=None),
            SimpleNamespace(id="d2", language="de", organization=None, organization_type=None,
                            region=None, cultural_sphere=None, publication_type=None,
                            country=None, publication_year=None),
            SimpleNamespace(id="d3", language="en", organization=None, organization_type=None,
                            region=None, cultural_sphere=None, publication_type=None,
                            country=None, publication_year=None),
        ]
        filtered = _apply_document_filters(docs, {"language": "en"})
        self.assertEqual([d.id for d in filtered], ["d1", "d3"])


class KwicQueryLanguageSchemaTests(unittest.TestCase):
    def test_query_language_alias_kwic_language(self) -> None:
        body = KwicRequest.model_validate(
            {
                "unit_type": "paragraph",
                "keyword": "policy",
                "query_mode": "lemma",
                "kwic_language": "en",
                "language": "de",
            }
        )
        self.assertEqual(body.query_language, "en")
        self.assertEqual(body.language, "de")

    def test_lemma_requires_query_language(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            KwicRequest.model_validate(
                {
                    "unit_type": "paragraph",
                    "keyword": "policy",
                    "query_mode": "lemma",
                }
            )
        self.assertIn("query_language", str(ctx.exception).lower())

    def test_legacy_language_mirrors_to_query_language_for_lemma(self) -> None:
        body = KwicRequest.model_validate(
            {
                "unit_type": "paragraph",
                "keyword": "policy",
                "query_mode": "lemma",
                "language": "en",
            }
        )
        self.assertEqual(body.language, "en")
        self.assertEqual(body.query_language, "en")

    def test_analysis_filters_keeps_language_excludes_query_language(self) -> None:
        body = KwicRequest.model_validate(
            {
                "unit_type": "paragraph",
                "keyword": "policy",
                "query_mode": "word",
                "language": "de",
                "query_language": "en",
            }
        )
        filters = _analysis_filters(body)
        self.assertEqual(filters.get("language"), "de")
        self.assertNotIn("query_language", filters)
        self.assertNotIn("keyword", filters)


class KwicServiceLanguageSeparationTests(unittest.IsolatedAsyncioTestCase):
    async def test_kwic_passes_corpus_and_query_language_independently(self) -> None:
        service = QuantitativeAnalysisService(db=MagicMock())
        corpus = SimpleNamespace(id="c1", project_id="p1")
        unit = SimpleNamespace(
            id="u1",
            text="Policies running.",
            corpus_document_id="d1",
            unit_type="paragraph",
            position=0,
            page_number=1,
            paragraph_number=1,
            sentence_number=None,
            char_start=0,
            char_end=17,
            section_heading=None,
        )
        documents = [
            SimpleNamespace(
                id="d1",
                title="Doc",
                organization=None,
                publication_year=None,
                source_url=None,
                language="de",
            )
        ]
        run = SimpleNamespace(id="run-1")

        service._select = AsyncMock(return_value=(corpus, [unit], documents))
        service._persist_run = AsyncMock(return_value=run)
        service._document_lookup = MagicMock(return_value={documents[0].id: documents[0]})
        service._resolve_config = AsyncMock(
            return_value=(
                SimpleNamespace(to_dict=lambda: {"language": "en"}),
                {"preprocessing_profile_id": None, "preprocessing_config": {}},
            )
        )

        with (
            patch(
                "backend.modules.text_research.application.quantitative_analysis_service._prepare_with_identity",
                new=AsyncMock(
                    return_value=(
                        SimpleNamespace(corpus_checksum="cc", pipeline_checksum="pc"),
                        {"scientific_identity": "x"},
                    )
                ),
            ),
            patch(
                "backend.modules.text_research.application.quantitative_analysis_service.quantitative.kwic_search",
                return_value=[{"keyword": "policy"}],
            ) as kwic_search,
        ):
            result = await service.kwic(
                "c1",
                user_id="user-1",
                unit_type="paragraph",
                keyword="policy",
                query_mode="lemma",
                query_language="en",
                language="de",
            )

        self.assertIs(result, run)
        select_filters = service._select.await_args.kwargs["filters"]
        self.assertEqual(select_filters.get("language"), "de")
        self.assertNotIn("query_language", select_filters)
        kwic_search.assert_called_once()
        self.assertEqual(kwic_search.call_args.kwargs["language"], "en")
        self.assertEqual(kwic_search.call_args.kwargs["query_mode"], "lemma")
        persisted = service._persist_run.await_args.kwargs["parameters"]
        self.assertEqual(persisted["query_language"], "en")
        self.assertEqual(persisted["filters"]["language"], "de")


if __name__ == "__main__":
    unittest.main()
