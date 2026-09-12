"""TASK-016: dictionary term override precedence and filter contract."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from backend.modules.text_research.api.schemas import CorpusFilters, DictionaryAnalysisRequest
from backend.modules.text_research.application.quantitative_analysis_service import (
    QuantitativeAnalysisService,
)
from backend.modules.text_research.infrastructure.dictionary_matcher import (
    DictionarySpec,
    parse_dictionary_payload,
)


class DictionaryPrecedenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_custom_terms_override_dictionary_id(self) -> None:
        service = QuantitativeAnalysisService(db=MagicMock())
        corpus = SimpleNamespace(id="c1", project_id="p1")
        units = [
            SimpleNamespace(
                id="u1",
                text="alpha beta gamma",
                corpus_document_id="d1",
                page_number=1,
                section_heading=None,
                char_start=0,
                char_end=10,
            )
        ]
        documents = [
            SimpleNamespace(id="d1", title="Doc", organization=None, publication_year=2020)
        ]
        stored = parse_dictionary_payload(["stored_only", "ignored"])
        dictionary = SimpleNamespace(
            id="dict-1",
            project_id="p1",
            name="Stored",
            version="1",
            description=None,
        )

        service._select = AsyncMock(return_value=(corpus, units, documents))
        service._resolve_config = AsyncMock(return_value=({}, {}))
        service._document_lookup = MagicMock(return_value={documents[0].id: documents[0]})
        captured: dict = {}

        async def _persist(_corpus, _run_type, **kwargs):
            captured["parameters"] = kwargs.get("parameters") or {}
            captured["results"] = kwargs.get("results") or {}
            return SimpleNamespace(id="run-1", status="completed")

        service._persist_run = AsyncMock(side_effect=_persist)

        def fake_match(tokenized, spec, **kwargs):
            assert isinstance(spec, DictionarySpec)
            captured["flat"] = spec.flattened_terms()
            return {
                "total_hits": 0,
                "normalized_hits": 0.0,
                "hits_per_1000_tokens": 0.0,
                "document_prevalence": 0.0,
                "matches": [],
                "per_unit": [{"hits": 0}],
                "dictionary": {},
            }

        with (
            patch(
                "backend.modules.text_research.application.quantitative_analysis_service._prepare_with_identity",
                new=AsyncMock(
                    return_value=(
                        SimpleNamespace(
                            corpus_checksum="c",
                            pipeline_checksum="p",
                            token_sequences=[["alpha", "beta", "gamma"]],
                            texts_joined=["alpha beta gamma"],
                            preprocessing_profile={},
                        ),
                        {},
                    )
                ),
            ),
            patch(
                "backend.modules.text_research.application.dictionary_service.DictionaryService.get_dictionary",
                new=AsyncMock(return_value=dictionary),
            ),
            patch(
                "backend.modules.text_research.application.dictionary_service.DictionaryService.get_spec",
                return_value=stored,
            ),
            patch(
                "backend.modules.text_research.infrastructure.dictionary_matcher.match_dictionary",
                side_effect=fake_match,
            ),
            patch(
                "backend.modules.text_research.application.quantitative_analysis_service._tokenized_from_prepared",
                return_value=[["alpha", "beta", "gamma"]],
            ),
            patch(
                "backend.modules.text_research.application.quantitative_analysis_service.asyncio.to_thread",
                new=AsyncMock(side_effect=lambda fn, *a, **k: fn()),
            ),
        ):
            await service.dictionary(
                "c1",
                user_id="user-1",
                unit_type="paragraph",
                dictionary_id="dict-1",
                dictionary_terms=["alpha", "custom_term"],
            )

        self.assertEqual(captured["flat"], ["alpha", "custom_term"])
        self.assertNotIn("stored_only", captured["flat"])
        self.assertEqual(
            captured["results"]["dictionary"]["terms_source"], "request_terms_override"
        )
        self.assertEqual(captured["results"]["dictionary"]["dictionary_id"], "dict-1")

    def test_schema_documents_override_semantics(self) -> None:
        req = DictionaryAnalysisRequest(
            unit_type="paragraph",
            dictionary_id="dict-1",
            dictionary_terms=["custom_a", "custom_b"],
        )
        description = DictionaryAnalysisRequest.model_fields["dictionary_terms"].description or ""
        self.assertIn("override", description.lower())


class FilterContractTests(unittest.TestCase):
    def test_country_and_exact_year_on_corpus_filters(self) -> None:
        filters = CorpusFilters(country="DE", publication_year=2021)
        dumped = filters.model_dump(exclude_none=True)
        self.assertEqual(dumped["country"], "DE")
        self.assertEqual(dumped["publication_year"], 2021)


if __name__ == "__main__":
    unittest.main()
