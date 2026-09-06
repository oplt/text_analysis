"""Quantitative text-analysis workflows: corpus stats, frequencies, n-grams,
DFM, KWIC, dictionary scoring, keyness, and co-occurrence.

Every method here computes real statistics from persisted `TextUnit` text via
`infrastructure.quantitative`, and persists an `AnalysisRun` recording the
parameters (including the preprocessing profile) and results for
reproducibility. Nothing is fabricated.
"""

from __future__ import annotations

import asyncio
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
from backend.modules.text_research.infrastructure.feature_cache import build_cache_key
from backend.modules.text_research.infrastructure.preprocessing import PreprocessingConfig


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _filter_kwargs(filters: dict[str, Any]) -> dict[str, Any]:
    """Map analysis filter names onto repository document-select kwargs."""
    mapping = {
        "organization": "organization",
        "organization_type": "organization_type",
        "region": "region",
        "cultural_sphere": "cultural_sphere",
        "language": "language",
        "publication_type": "publication_type",
        "country": "country",
        "publication_year": "publication_year",
        "publication_year_min": "publication_year_min",
        "publication_year_max": "publication_year_max",
        "year_min": "publication_year_min",
        "year_max": "publication_year_max",
    }
    return {
        mapping[key]: value
        for key, value in filters.items()
        if key in mapping and value is not None
    }


def _apply_document_filters(
    documents: list[CorpusDocument], filters: dict[str, Any]
) -> list[CorpusDocument]:
    """Legacy in-memory filter kept for callers that already loaded documents."""
    result = documents
    kwargs = _filter_kwargs(filters)
    if kwargs.get("organization"):
        result = [d for d in result if d.organization == kwargs["organization"]]
    if kwargs.get("organization_type"):
        result = [d for d in result if d.organization_type == kwargs["organization_type"]]
    if kwargs.get("region"):
        result = [d for d in result if d.region == kwargs["region"]]
    if kwargs.get("cultural_sphere"):
        result = [d for d in result if d.cultural_sphere == kwargs["cultural_sphere"]]
    if kwargs.get("language"):
        result = [d for d in result if d.language == kwargs["language"]]
    if kwargs.get("publication_type"):
        result = [d for d in result if d.publication_type == kwargs["publication_type"]]
    if kwargs.get("country"):
        result = [d for d in result if d.country == kwargs["country"]]
    if kwargs.get("publication_year") is not None:
        result = [d for d in result if d.publication_year == kwargs["publication_year"]]
    if kwargs.get("publication_year_min") is not None:
        result = [
            d
            for d in result
            if d.publication_year is not None
            and d.publication_year >= kwargs["publication_year_min"]
        ]
    if kwargs.get("publication_year_max") is not None:
        result = [
            d
            for d in result
            if d.publication_year is not None
            and d.publication_year <= kwargs["publication_year_max"]
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
        filter_kwargs = _filter_kwargs(filters or {})
        documents = await self.repo.list_documents(corpus_id, **filter_kwargs)
        if filters and filters.get("document_ids"):
            wanted = set(filters["document_ids"])
            documents = [d for d in documents if d.id in wanted]
        doc_ids = [d.id for d in documents]
        units = await self.repo.list_text_units_for_corpus(
            corpus_id,
            unit_type=unit_type,
            document_ids=doc_ids if (filters and filter_kwargs) or (filters and filters.get("document_ids")) else None,
        )
        if not units:
            raise HTTPException(
                status_code=422,
                detail="No text units match the requested corpus/unit_type/filters. "
                "Segment the corpus first or relax filters.",
            )
        return corpus, units, documents

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

    def _cache_key(
        self,
        *,
        corpus_id: str,
        unit_type: str,
        units: list[TextUnit],
        config: PreprocessingConfig,
        filters: dict[str, Any] | None,
        mode: str,
    ) -> str:
        return build_cache_key(
            corpus_id=corpus_id,
            unit_type=unit_type,
            unit_ids=[u.id for u in units],
            unit_hashes=[u.text_hash for u in units],
            config=config,
            mode=mode,
            filters=filters,
        )

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
        cache_key = self._cache_key(
            corpus_id=corpus.id,
            unit_type=unit_type,
            units=units,
            config=config,
            filters=filters,
            mode="tokens",
        )
        stats = await asyncio.to_thread(
            quantitative.corpus_stats, texts, config, cache_key=cache_key
        )

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
        cache_key = self._cache_key(
            corpus_id=corpus.id,
            unit_type=unit_type,
            units=units,
            config=config,
            filters=filters,
            mode="tokens",
        )
        rows = await asyncio.to_thread(
            quantitative.compute_frequencies,
            [u.text for u in units],
            config,
            top_n=top_n,
            cache_key=cache_key,
        )
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
        cache_key = self._cache_key(
            corpus_id=corpus.id, unit_type=unit_type, units=units, config=config, filters=filters, mode="tokens"
        )
        rows = await asyncio.to_thread(
            quantitative.compute_ngrams,
            [u.text for u in units],
            config,
            n=n,
            top_n=top_n,
            cache_key=cache_key,
        )
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
        cache_key = self._cache_key(
            corpus_id=corpus.id, unit_type=unit_type, units=units, config=config, filters=filters, mode="tokens"
        )
        result = await asyncio.to_thread(
            quantitative.build_dfm,
            [u.text for u in units],
            [u.id for u in units],
            config,
            weighting=weighting,
            cache_key=cache_key,
        )
        summary = await asyncio.to_thread(quantitative.dfm_summary, result)
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
        matches = await asyncio.to_thread(
            quantitative.kwic_search,
            payload,
            keyword,
            window_size=window_size,
            case_sensitive=case_sensitive,
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
        cache_key = self._cache_key(
            corpus_id=corpus.id, unit_type=unit_type, units=units, config=config, filters=filters, mode="tokens"
        )
        doc_lookup = self._document_lookup(units, documents)
        group_keys = None
        if group_by:
            group_keys = [
                str(getattr(doc_lookup.get(u.corpus_document_id), group_by, None) or "unspecified")
                for u in units
            ]
        result = await asyncio.to_thread(
            quantitative.dictionary_analysis,
            [u.text for u in units],
            [u.id for u in units],
            dictionary_terms,
            config,
            group_keys=group_keys,
            cache_key=cache_key,
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
        cache_key_a = self._cache_key(
            corpus_id=corpus.id, unit_type=unit_type, units=units_a, config=config, filters=filters_a, mode="tokens"
        )
        cache_key_b = self._cache_key(
            corpus_id=corpus.id, unit_type=unit_type, units=units_b, config=config, filters=filters_b, mode="tokens"
        )
        rows = await asyncio.to_thread(
            quantitative.keyness_for_texts,
            [u.text for u in units_a],
            [u.text for u in units_b],
            config,
            top_n=top_n,
            cache_key_a=cache_key_a,
            cache_key_b=cache_key_b,
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
        cache_key = self._cache_key(
            corpus_id=corpus.id, unit_type=unit_type, units=units, config=config, filters=filters, mode="tokens"
        )
        rows = await asyncio.to_thread(
            quantitative.cooccurrence_for_texts,
            [u.text for u in units],
            config,
            window_size=window_size,
            top_n=top_n,
            cache_key=cache_key,
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
