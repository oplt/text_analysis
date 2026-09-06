"""Quantitative text-analysis workflows: corpus stats, frequencies, n-grams,
DFM, KWIC, dictionary scoring, keyness, and co-occurrence.

Every method here computes real statistics from persisted `TextUnit` text via
`infrastructure.quantitative`, and persists an `AnalysisRun` recording the
parameters (including the preprocessing profile) and results for
reproducibility. Nothing is fabricated.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import HTTPException

from backend.modules.text_research.application.access import ResearchAccessMixin
from backend.modules.text_research.application.preprocessing_service import (
    PreprocessingProfileService,
)
from backend.modules.text_research.domain.enums import AnalysisRunStatus, AnalysisRunType
from backend.modules.text_research.domain.models import AnalysisRun, CorpusDocument, TextUnit, dumps
from backend.modules.text_research.infrastructure import quantitative
from backend.modules.text_research.infrastructure.preprocessing import PreprocessingConfig


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _apply_document_filters(
    documents: list[CorpusDocument], filters: dict[str, Any]
) -> list[CorpusDocument]:
    result = documents
    if filters.get("organization"):
        result = [d for d in result if d.organization == filters["organization"]]
    if filters.get("region"):
        result = [d for d in result if d.region == filters["region"]]
    if filters.get("cultural_sphere"):
        result = [d for d in result if d.cultural_sphere == filters["cultural_sphere"]]
    if filters.get("language"):
        result = [d for d in result if d.language == filters["language"]]
    if filters.get("publication_type"):
        result = [d for d in result if d.publication_type == filters["publication_type"]]
    if filters.get("year_min") is not None:
        result = [
            d for d in result if d.publication_year is not None and d.publication_year >= filters["year_min"]
        ]
    if filters.get("year_max") is not None:
        result = [
            d for d in result if d.publication_year is not None and d.publication_year <= filters["year_max"]
        ]
    if filters.get("document_ids"):
        wanted = set(filters["document_ids"])
        result = [d for d in result if d.id in wanted]
    return result


class QuantitativeAnalysisService(ResearchAccessMixin):
    async def _resolve_config(
        self, preprocessing_profile_id: str | None, *, user_id: str
    ) -> tuple[PreprocessingConfig, dict[str, Any]]:
        if preprocessing_profile_id is None:
            config = PreprocessingConfig()
            return config, {"preprocessing_profile_id": None, "preprocessing_config": config.to_dict()}
        profile = await self.get_preprocessing_profile_or_404(preprocessing_profile_id, user_id=user_id)
        config = PreprocessingProfileService.resolve_config(profile)
        return config, {
            "preprocessing_profile_id": profile.id,
            "preprocessing_config": config.to_dict(),
        }

    async def _select(
        self, corpus_id: str, *, user_id: str, unit_type: str, filters: dict[str, Any] | None = None
    ) -> tuple[Any, list[TextUnit], list[CorpusDocument]]:
        corpus = await self.get_corpus_or_404(corpus_id, user_id=user_id)
        documents = await self.repo.list_documents(corpus_id)
        filtered_docs = _apply_document_filters(documents, filters or {})
        doc_ids = [d.id for d in filtered_docs]
        units = await self.repo.list_text_units_for_corpus(
            corpus_id, unit_type=unit_type, document_ids=doc_ids if filters else None
        )
        if not units:
            raise HTTPException(
                status_code=422,
                detail="No text units match the requested corpus/unit_type/filters. "
                "Segment the corpus first or relax filters.",
            )
        return corpus, units, filtered_docs

    async def _persist_run(
        self,
        corpus,
        run_type: AnalysisRunType,
        *,
        user_id: str,
        parameters: dict[str, Any],
        metrics: dict[str, Any],
        results: dict[str, Any],
    ) -> AnalysisRun:
        run = await self.repo.create_run(
            AnalysisRun(
                project_id=corpus.project_id,
                corpus_id=corpus.id,
                run_type=run_type.value,
                status=AnalysisRunStatus.COMPLETED.value,
                parameters_json=dumps(parameters),
                metrics_json=dumps(metrics),
                results_json=dumps(results),
                created_by=user_id,
                started_at=_utcnow(),
                completed_at=_utcnow(),
            )
        )
        await self.db.commit()
        return run

    def _document_lookup(self, units: list[TextUnit], documents: list[CorpusDocument]) -> dict[str, CorpusDocument]:
        return {d.id: d for d in documents}

    # ------------------------------------------------------------------
    async def corpus_stats(
        self,
        corpus_id: str,
        *,
        user_id: str,
        unit_type: str,
        preprocessing_profile_id: str | None = None,
        **filters: Any,
    ) -> AnalysisRun:
        corpus, units, documents = await self._select(
            corpus_id, user_id=user_id, unit_type=unit_type, filters=filters
        )
        config, config_params = await self._resolve_config(preprocessing_profile_id, user_id=user_id)
        texts = [u.text for u in units]
        stats = quantitative.corpus_stats(texts, config)

        doc_lookup = self._document_lookup(units, documents)

        def group_counts(attr: str) -> dict[str, int]:
            counts: dict[str, int] = {}
            for unit in units:
                doc = doc_lookup.get(unit.corpus_document_id)
                key = str(getattr(doc, attr, None) or "unspecified") if doc else "unspecified"
                counts[key] = counts.get(key, 0) + 1
            return counts

        breakdowns = {
            "by_organization": group_counts("organization"),
            "by_year": group_counts("publication_year"),
            "by_region": group_counts("region"),
            "by_cultural_sphere": group_counts("cultural_sphere"),
            "by_language": group_counts("language"),
        }
        return await self._persist_run(
            corpus,
            AnalysisRunType.CORPUS_STATS,
            user_id=user_id,
            parameters={"unit_type": unit_type, "filters": filters, **config_params},
            metrics=stats,
            results={"breakdowns": breakdowns},
        )

    async def frequencies(
        self,
        corpus_id: str,
        *,
        user_id: str,
        unit_type: str,
        preprocessing_profile_id: str | None = None,
        top_n: int = 50,
        **filters: Any,
    ) -> AnalysisRun:
        corpus, units, _ = await self._select(corpus_id, user_id=user_id, unit_type=unit_type, filters=filters)
        config, config_params = await self._resolve_config(preprocessing_profile_id, user_id=user_id)
        rows = quantitative.compute_frequencies([u.text for u in units], config, top_n=top_n)
        return await self._persist_run(
            corpus,
            AnalysisRunType.FREQUENCY_ANALYSIS,
            user_id=user_id,
            parameters={"unit_type": unit_type, "top_n": top_n, "filters": filters, **config_params},
            metrics={"unit_count": len(units), "unique_terms_returned": len(rows)},
            results={"frequencies": rows},
        )

    async def ngrams(
        self,
        corpus_id: str,
        *,
        user_id: str,
        unit_type: str,
        n: int = 2,
        preprocessing_profile_id: str | None = None,
        top_n: int = 50,
        **filters: Any,
    ) -> AnalysisRun:
        corpus, units, _ = await self._select(corpus_id, user_id=user_id, unit_type=unit_type, filters=filters)
        config, config_params = await self._resolve_config(preprocessing_profile_id, user_id=user_id)
        rows = quantitative.compute_ngrams([u.text for u in units], config, n=n, top_n=top_n)
        return await self._persist_run(
            corpus,
            AnalysisRunType.NGRAM_ANALYSIS,
            user_id=user_id,
            parameters={"unit_type": unit_type, "n": n, "top_n": top_n, "filters": filters, **config_params},
            metrics={"unit_count": len(units), "ngrams_returned": len(rows)},
            results={"ngrams": rows},
        )

    async def dfm(
        self,
        corpus_id: str,
        *,
        user_id: str,
        unit_type: str,
        weighting: str = "count",
        preprocessing_profile_id: str | None = None,
        **filters: Any,
    ) -> AnalysisRun:
        corpus, units, _ = await self._select(corpus_id, user_id=user_id, unit_type=unit_type, filters=filters)
        config, config_params = await self._resolve_config(preprocessing_profile_id, user_id=user_id)
        result = quantitative.build_dfm(
            [u.text for u in units], [u.id for u in units], config, weighting=weighting
        )
        summary = quantitative.dfm_summary(result)
        return await self._persist_run(
            corpus,
            AnalysisRunType.DFM,
            user_id=user_id,
            parameters={"unit_type": unit_type, "weighting": weighting, "filters": filters, **config_params},
            metrics={"unit_count": summary["unit_count"], "feature_count": summary["feature_count"], "density": summary["density"]},
            results={"summary": summary},
        )

    async def kwic(
        self,
        corpus_id: str,
        *,
        user_id: str,
        unit_type: str,
        keyword: str,
        window_size: int = 5,
        case_sensitive: bool = False,
        **filters: Any,
    ) -> AnalysisRun:
        corpus, units, documents = await self._select(corpus_id, user_id=user_id, unit_type=unit_type, filters=filters)
        doc_lookup = self._document_lookup(units, documents)
        payload = []
        for unit in units:
            doc = doc_lookup.get(unit.corpus_document_id)
            payload.append(
                {
                    "id": unit.id,
                    "text": unit.text,
                    "document_title": doc.title if doc else None,
                    "organization": doc.organization if doc else None,
                    "publication_year": doc.publication_year if doc else None,
                }
            )
        matches = quantitative.kwic_search(
            payload, keyword, window_size=window_size, case_sensitive=case_sensitive
        )
        return await self._persist_run(
            corpus,
            AnalysisRunType.KWIC,
            user_id=user_id,
            parameters={
                "unit_type": unit_type,
                "keyword": keyword,
                "window_size": window_size,
                "case_sensitive": case_sensitive,
                "filters": filters,
            },
            metrics={"unit_count": len(units), "match_count": len(matches)},
            results={"matches": matches},
        )

    async def dictionary(
        self,
        corpus_id: str,
        *,
        user_id: str,
        unit_type: str,
        dictionary_terms: list[str],
        dictionary_id: str | None = None,
        group_by: str | None = None,
        preprocessing_profile_id: str | None = None,
        **filters: Any,
    ) -> AnalysisRun:
        corpus, units, documents = await self._select(corpus_id, user_id=user_id, unit_type=unit_type, filters=filters)
        config, config_params = await self._resolve_config(preprocessing_profile_id, user_id=user_id)
        doc_lookup = self._document_lookup(units, documents)
        group_keys = None
        if group_by:
            group_keys = [
                str(getattr(doc_lookup.get(u.corpus_document_id), group_by, None) or "unspecified")
                for u in units
            ]
        result = quantitative.dictionary_analysis(
            [u.text for u in units], [u.id for u in units], dictionary_terms, config, group_keys=group_keys
        )
        return await self._persist_run(
            corpus,
            AnalysisRunType.DICTIONARY_ANALYSIS,
            user_id=user_id,
            parameters={
                "unit_type": unit_type,
                "dictionary_id": dictionary_id,
                "dictionary_terms": dictionary_terms,
                "group_by": group_by,
                "filters": filters,
                **config_params,
            },
            metrics={
                "total_hits": result["total_hits"],
                "hits_per_1000_tokens": result["hits_per_1000_tokens"],
                "document_prevalence": result["document_prevalence"],
            },
            results=result,
        )

    async def keyness(
        self,
        corpus_id: str,
        *,
        user_id: str,
        unit_type: str,
        filters_a: dict[str, Any],
        filters_b: dict[str, Any],
        preprocessing_profile_id: str | None = None,
        top_n: int = 50,
    ) -> AnalysisRun:
        corpus, units_a, _ = await self._select(corpus_id, user_id=user_id, unit_type=unit_type, filters=filters_a)
        _, units_b, _ = await self._select(corpus_id, user_id=user_id, unit_type=unit_type, filters=filters_b)
        config, config_params = await self._resolve_config(preprocessing_profile_id, user_id=user_id)
        rows = quantitative.keyness_for_texts(
            [u.text for u in units_a], [u.text for u in units_b], config, top_n=top_n
        )
        return await self._persist_run(
            corpus,
            AnalysisRunType.KEYNESS,
            user_id=user_id,
            parameters={
                "unit_type": unit_type,
                "filters_a": filters_a,
                "filters_b": filters_b,
                "top_n": top_n,
                **config_params,
            },
            metrics={"unit_count_a": len(units_a), "unit_count_b": len(units_b), "features_returned": len(rows)},
            results={"keyness": rows},
        )

    async def cooccurrence(
        self,
        corpus_id: str,
        *,
        user_id: str,
        unit_type: str,
        window_size: int = 5,
        top_n: int = 50,
        preprocessing_profile_id: str | None = None,
        **filters: Any,
    ) -> AnalysisRun:
        corpus, units, _ = await self._select(corpus_id, user_id=user_id, unit_type=unit_type, filters=filters)
        config, config_params = await self._resolve_config(preprocessing_profile_id, user_id=user_id)
        rows = quantitative.cooccurrence_for_texts(
            [u.text for u in units], config, window_size=window_size, top_n=top_n
        )
        return await self._persist_run(
            corpus,
            AnalysisRunType.COOCCURRENCE,
            user_id=user_id,
            parameters={
                "unit_type": unit_type,
                "window_size": window_size,
                "top_n": top_n,
                "filters": filters,
                **config_params,
            },
            metrics={"unit_count": len(units), "pairs_returned": len(rows)},
            results={"cooccurrence": rows},
        )
