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
from backend.modules.text_research.infrastructure.preprocessing import (
    PreprocessingConfig,
    describe_implementation,
)


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
            return config, {
                "preprocessing_profile_id": None,
                "preprocessing_config": config.to_dict(),
                "preprocessing_implementation": describe_implementation(config),
            }
        profile = await self.get_preprocessing_profile_or_404(preprocessing_profile_id, user_id=user_id)
        config = PreprocessingProfileService.resolve_config(profile)
        return config, {
            "preprocessing_profile_id": profile.id,
            "preprocessing_config": config.to_dict(),
            "preprocessing_implementation": describe_implementation(config),
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
            quantitative.corpus_stats,
            texts,
            config,
            cache_key=cache_key,
            document_ids=[u.corpus_document_id for u in units],
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
            parameters={
                "unit_type": unit_type,
                "filters": filters,
                "pipeline_workflow": list(quantitative.PIPELINE_STAGES),
                **config_params,
            },
            metrics=stats,
            results={
                "breakdowns": breakdowns,
                "lexical_diversity": stats.get("lexical_diversity"),
                "pipeline": stats.get("pipeline"),
            },
        )

    async def frequencies(
        self,
        corpus_id: str,
        *,
        user_id: str,
        unit_type: str,
        preprocessing_profile_id: str | None = None,
        top_n: int = 50,
        rate_per: float = 1000,
        group_by: str | None = None,
        **filters: Any,
    ) -> AnalysisRun:
        corpus, units, documents = await self._select(
            corpus_id, user_id=user_id, unit_type=unit_type, filters=filters
        )
        config, config_params = await self._resolve_config(preprocessing_profile_id, user_id=user_id)
        cache_key = self._cache_key(
            corpus_id=corpus.id,
            unit_type=unit_type,
            units=units,
            config=config,
            filters=filters,
            mode="tokens",
        )
        doc_lookup = self._document_lookup(units, documents)
        group_keys: list[str] | None = None
        if group_by:
            allowed = {
                "organization",
                "organization_type",
                "publication_year",
                "region",
                "cultural_sphere",
                "language",
                "publication_type",
                "country",
            }
            if group_by not in allowed:
                raise HTTPException(
                    status_code=422,
                    detail=f"Unsupported group_by {group_by!r}; expected one of {sorted(allowed)}",
                )
            group_keys = []
            for unit in units:
                doc = doc_lookup.get(unit.corpus_document_id)
                value = getattr(doc, group_by, None) if doc else None
                group_keys.append(str(value) if value is not None else "unspecified")

        report = await asyncio.to_thread(
            quantitative.compute_frequencies,
            [u.text for u in units],
            config,
            top_n=top_n,
            rate_per=rate_per,
            cache_key=cache_key,
            unit_ids=[u.id for u in units],
            document_ids=[u.corpus_document_id for u in units],
            group_keys=group_keys,
        )
        rows = report["frequencies"]
        metadata = report["metadata"]
        return await self._persist_run(
            corpus,
            AnalysisRunType.FREQUENCY_ANALYSIS,
            user_id=user_id,
            parameters={
                "unit_type": unit_type,
                "top_n": top_n,
                "rate_per": rate_per,
                "group_by": group_by,
                "filters": filters,
                **config_params,
            },
            metrics={
                "unit_count": metadata["unit_count"],
                "token_count": metadata["token_count"],
                "vocabulary_size": metadata["vocabulary_size"],
                "unique_terms_returned": metadata["terms_returned"],
                "rate_per": metadata["rate_per"],
            },
            results={"frequencies": rows, "metadata": metadata},
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
        rate_per: float = 1000,
        skip: int = 0,
        **filters: Any,
    ) -> AnalysisRun:
        try:
            quantitative.validate_ngram_order(n, skip=skip)
            quantitative.resolve_rate_per(rate_per)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        corpus, units, _ = await self._select(
            corpus_id, user_id=user_id, unit_type=unit_type, filters=filters
        )
        config, config_params = await self._resolve_config(preprocessing_profile_id, user_id=user_id)
        cache_key = self._cache_key(
            corpus_id=corpus.id,
            unit_type=unit_type,
            units=units,
            config=config,
            filters=filters,
            mode="tokens",
        )
        report = await asyncio.to_thread(
            quantitative.compute_ngrams,
            [u.text for u in units],
            config,
            n=n,
            top_n=top_n,
            rate_per=rate_per,
            skip=skip,
            cache_key=cache_key,
            unit_ids=[u.id for u in units],
            document_ids=[u.corpus_document_id for u in units],
        )
        rows = report["ngrams"]
        metadata = report["metadata"]
        return await self._persist_run(
            corpus,
            AnalysisRunType.NGRAM_ANALYSIS,
            user_id=user_id,
            parameters={
                "unit_type": unit_type,
                "n": n,
                "top_n": top_n,
                "rate_per": rate_per,
                "skip": skip,
                "filters": filters,
                **config_params,
            },
            metrics={
                "unit_count": metadata["unit_count"],
                "ngram_token_count": metadata["ngram_token_count"],
                "vocabulary_size": metadata["vocabulary_size"],
                "ngrams_returned": metadata["ngrams_returned"],
                "n": metadata["n"],
                "n_label": metadata["n_label"],
            },
            results={"ngrams": rows, "metadata": metadata},
        )

    async def dfm(
        self,
        corpus_id: str,
        *,
        user_id: str,
        unit_type: str,
        weighting: str = "count",
        k1: float | None = None,
        b: float | None = None,
        smooth_idf: bool | None = None,
        preprocessing_profile_id: str | None = None,
        force_sparse_only: bool = False,
        trim: dict[str, Any] | None = None,
        **filters: Any,
    ) -> AnalysisRun:
        try:
            weighting = quantitative.normalize_dfm_weighting(weighting)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        corpus, units, _ = await self._select(
            corpus_id, user_id=user_id, unit_type=unit_type, filters=filters
        )
        config, config_params = await self._resolve_config(preprocessing_profile_id, user_id=user_id)
        cache_key = self._cache_key(
            corpus_id=corpus.id,
            unit_type=unit_type,
            units=units,
            config=config,
            filters=filters,
            mode="tokens",
        )
        build_kwargs: dict[str, Any] = {
            "weighting": weighting,
            "cache_key": cache_key,
            "force_sparse_only": force_sparse_only,
        }
        if k1 is not None:
            build_kwargs["k1"] = k1
        if b is not None:
            build_kwargs["b"] = b
        if smooth_idf is not None:
            build_kwargs["smooth_idf"] = smooth_idf
        if trim:
            build_kwargs["trim"] = {k: v for k, v in trim.items() if v is not None}

        try:
            result = await asyncio.to_thread(
                quantitative.build_dfm,
                [u.text for u in units],
                [u.id for u in units],
                config,
                **build_kwargs,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        summary = quantitative.dfm_summary(result)
        # Persist sparse DFM + provenance; omit dense_matrix from DB payload when present
        # to keep storage lean (dense remains available via preview / recompute).
        persisted = {
            "dimensions": result["dimensions"],
            "density": result["density"],
            "nnz": result["nnz"],
            "feature_names": result["feature_names"],
            "unit_ids": result["unit_ids"],
            "preprocessing_config": result.get("preprocessing_config"),
            "mode": result["mode"],
            "weighting": result["weighting"],
            "weighting_scheme": result.get("weighting_scheme"),
            "sublinear_tf": result.get("sublinear_tf", False),
            "sparse": result["sparse"],
            "storage": result["storage"],
            "preview": result.get("preview"),
            "trim": result.get("trim"),
        }
        return await self._persist_run(
            corpus,
            AnalysisRunType.DFM,
            user_id=user_id,
            parameters={
                "unit_type": unit_type,
                "weighting": weighting,
                "k1": k1,
                "b": b,
                "smooth_idf": smooth_idf,
                "force_sparse_only": force_sparse_only,
                "trim": trim,
                "filters": filters,
                **config_params,
            },
            metrics={
                "unit_count": summary["unit_count"],
                "feature_count": summary["feature_count"],
                "density": summary["density"],
                "nnz": summary["nnz"],
                "storage": summary["storage"],
                "mode": summary["mode"],
                "trim": summary.get("trim"),
            },
            results={"summary": summary, "dfm": persisted},
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
        query_mode: str = "auto",
        language: str | None = None,
        token_attribute: str | None = None,
        max_matches: int | None = None,
        **filters: Any,
    ) -> AnalysisRun:
        corpus, units, documents = await self._select(
            corpus_id, user_id=user_id, unit_type=unit_type, filters=filters
        )
        doc_lookup = self._document_lookup(units, documents)
        payload = []
        for unit in units:
            doc = doc_lookup.get(unit.corpus_document_id)
            payload.append(
                {
                    "text": unit.text,
                    "text_unit_id": unit.id,
                    "id": unit.id,
                    "corpus_document_id": unit.corpus_document_id,
                    "unit_type": unit.unit_type,
                    "position": unit.position,
                    "page_number": unit.page_number,
                    "paragraph_number": unit.paragraph_number,
                    "sentence_number": unit.sentence_number,
                    "unit_char_start": unit.char_start,
                    "unit_char_end": unit.char_end,
                    "section_heading": unit.section_heading,
                    "document_title": doc.title if doc else None,
                    "organization": doc.organization if doc else None,
                    "publication_year": doc.publication_year if doc else None,
                    "source_url": doc.source_url if doc else None,
                }
            )
        try:
            matches = await asyncio.to_thread(
                quantitative.kwic_search,
                payload,
                keyword,
                window_size=window_size,
                case_sensitive=case_sensitive,
                query_mode=query_mode,
                language=language,
                token_attribute=token_attribute,
                max_matches=max_matches,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return await self._persist_run(
            corpus,
            AnalysisRunType.KWIC,
            user_id=user_id,
            parameters={
                "unit_type": unit_type,
                "keyword": keyword,
                "window_size": window_size,
                "case_sensitive": case_sensitive,
                "query_mode": query_mode,
                "language": language,
                "token_attribute": token_attribute,
                "max_matches": max_matches,
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
        dictionary_terms: list[str] | None = None,
        dictionary_id: str | None = None,
        hierarchy: dict[str, Any] | None = None,
        exclusions: list[Any] | None = None,
        dictionary_language: str | None = None,
        case_sensitive: bool = False,
        rate_per: float = 1000.0,
        group_by: str | None = None,
        preprocessing_profile_id: str | None = None,
        **filters: Any,
    ) -> AnalysisRun:
        from backend.modules.text_research.application.dictionary_service import DictionaryService
        from backend.modules.text_research.infrastructure.dictionary_matcher import (
            parse_dictionary_payload,
        )

        corpus, units, documents = await self._select(
            corpus_id, user_id=user_id, unit_type=unit_type, filters=filters
        )
        config, config_params = await self._resolve_config(preprocessing_profile_id, user_id=user_id)
        cache_key = self._cache_key(
            corpus_id=corpus.id,
            unit_type=unit_type,
            units=units,
            config=config,
            filters=filters,
            mode="tokens",
        )
        doc_lookup = self._document_lookup(units, documents)

        dictionary_meta: dict[str, Any] = {"source": "user"}
        if dictionary_id:
            dictionary = await DictionaryService(self.db).get_dictionary(
                dictionary_id, user_id=user_id
            )
            if dictionary.project_id != corpus.project_id:
                raise HTTPException(
                    status_code=400,
                    detail="Dictionary does not belong to this corpus project",
                )
            spec = DictionaryService.get_spec(dictionary)
            if dictionary_language:
                spec.language = dictionary_language
            dictionary_meta.update(
                {
                    "dictionary_id": dictionary.id,
                    "name": dictionary.name,
                    "version": dictionary.version,
                    "description": dictionary.description,
                }
            )
        elif hierarchy is not None:
            spec = parse_dictionary_payload(
                {
                    "hierarchy": hierarchy,
                    "exclusions": exclusions or [],
                    "language": dictionary_language,
                    "source": "user",
                },
                language=dictionary_language,
            )
        else:
            terms = dictionary_terms or []
            if not terms:
                raise HTTPException(
                    status_code=400,
                    detail="dictionary_terms or hierarchy required when dictionary_id is omitted",
                )
            spec = parse_dictionary_payload(
                {
                    "terms": terms,
                    "exclusions": exclusions or [],
                    "language": dictionary_language,
                    "source": "user",
                },
                language=dictionary_language,
            )

        group_keys = None
        if group_by:
            group_keys = [
                str(getattr(doc_lookup.get(u.corpus_document_id), group_by, None) or "unspecified")
                for u in units
            ]

        unit_metadata = []
        for unit in units:
            doc = doc_lookup.get(unit.corpus_document_id)
            unit_metadata.append(
                {
                    "text_unit_id": unit.id,
                    "corpus_document_id": unit.corpus_document_id,
                    "page_number": unit.page_number,
                    "section_heading": unit.section_heading,
                    "unit_char_start": unit.char_start,
                    "unit_char_end": unit.char_end,
                    "document_title": doc.title if doc else None,
                    "organization": doc.organization if doc else None,
                    "publication_year": doc.publication_year if doc else None,
                }
            )

        try:
            result = await asyncio.to_thread(
                quantitative.dictionary_analysis,
                [u.text for u in units],
                [u.id for u in units],
                spec,
                config,
                group_keys=group_keys,
                cache_key=cache_key,
                metadata=unit_metadata,
                case_sensitive=case_sensitive,
                rate_per=rate_per,
                dictionary_meta=dictionary_meta,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return await self._persist_run(
            corpus,
            AnalysisRunType.DICTIONARY_ANALYSIS,
            user_id=user_id,
            parameters={
                "unit_type": unit_type,
                "dictionary_id": dictionary_id,
                "dictionary_terms": dictionary_terms,
                "hierarchy": hierarchy,
                "exclusions": exclusions,
                "dictionary_language": dictionary_language or spec.language,
                "case_sensitive": case_sensitive,
                "rate_per": rate_per,
                "group_by": group_by,
                "filters": filters,
                **config_params,
            },
            metrics={
                "total_hits": result["total_hits"],
                "normalized_hits": result["normalized_hits"],
                "hits_per_1000_tokens": result["hits_per_1000_tokens"],
                "document_prevalence": result["document_prevalence"],
                "match_count": len(result.get("matches") or []),
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
        group_field: str | None = None,
        method: str = "log_likelihood",
        correction: str = "bh",
        min_frequency: int = 1,
        preprocessing_profile_id: str | None = None,
        top_n: int = 50,
    ) -> AnalysisRun:
        if not filters_a or not filters_b:
            raise HTTPException(
                status_code=400,
                detail="filters_a and filters_b are required — choose comparison groups from corpus metadata",
            )
        corpus, units_a, _ = await self._select(
            corpus_id, user_id=user_id, unit_type=unit_type, filters=filters_a
        )
        _, units_b, _ = await self._select(
            corpus_id, user_id=user_id, unit_type=unit_type, filters=filters_b
        )
        if not units_a or not units_b:
            raise HTTPException(
                status_code=400,
                detail="Both comparison groups must contain at least one text unit",
            )
        config, config_params = await self._resolve_config(preprocessing_profile_id, user_id=user_id)
        cache_key_a = self._cache_key(
            corpus_id=corpus.id,
            unit_type=unit_type,
            units=units_a,
            config=config,
            filters=filters_a,
            mode="tokens",
        )
        cache_key_b = self._cache_key(
            corpus_id=corpus.id,
            unit_type=unit_type,
            units=units_b,
            config=config,
            filters=filters_b,
            mode="tokens",
        )

        inferred_field = group_field
        if inferred_field is None:
            shared = set(filters_a) & set(filters_b)
            if len(shared) == 1:
                inferred_field = next(iter(shared))

        label_a = ", ".join(f"{k}={v}" for k, v in sorted(filters_a.items()))
        label_b = ", ".join(f"{k}={v}" for k, v in sorted(filters_b.items()))

        try:
            report = await asyncio.to_thread(
                quantitative.keyness_for_texts,
                [u.text for u in units_a],
                [u.text for u in units_b],
                config,
                top_n=top_n,
                method=method,
                min_frequency=min_frequency,
                correction=correction,
                group_a_label=label_a,
                group_b_label=label_b,
                group_field=inferred_field,
                cache_key_a=cache_key_a,
                cache_key_b=cache_key_b,
                as_report=True,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        rows = report["features"]
        return await self._persist_run(
            corpus,
            AnalysisRunType.KEYNESS,
            user_id=user_id,
            parameters={
                "unit_type": unit_type,
                "filters_a": filters_a,
                "filters_b": filters_b,
                "group_field": inferred_field,
                "method": report["method"],
                "correction": report["correction"],
                "min_frequency": min_frequency,
                "top_n": top_n,
                **config_params,
            },
            metrics={
                "unit_count_a": len(units_a),
                "unit_count_b": len(units_b),
                "features_tested": report["features_tested"],
                "features_returned": len(rows),
                "method": report["method"],
                "correction": report["correction"],
            },
            results={"keyness": rows, "report": report},
        )

    async def cooccurrence(
        self,
        corpus_id: str,
        *,
        user_id: str,
        unit_type: str,
        window_size: int = 5,
        top_n: int = 50,
        association_method: str = "pmi",
        directional: bool = False,
        min_frequency: int = 1,
        min_count: int = 1,
        include_network: bool = True,
        preprocessing_profile_id: str | None = None,
        **filters: Any,
    ) -> AnalysisRun:
        corpus, units, _ = await self._select(
            corpus_id, user_id=user_id, unit_type=unit_type, filters=filters
        )
        config, config_params = await self._resolve_config(preprocessing_profile_id, user_id=user_id)
        cache_key = self._cache_key(
            corpus_id=corpus.id,
            unit_type=unit_type,
            units=units,
            config=config,
            filters=filters,
            mode="tokens",
        )
        try:
            report = await asyncio.to_thread(
                quantitative.cooccurrence_for_texts,
                [u.text for u in units],
                config,
                window_size=window_size,
                top_n=top_n,
                association_method=association_method,
                directional=directional,
                min_frequency=min_frequency,
                min_count=min_count,
                include_network=include_network,
                cache_key=cache_key,
                as_report=True,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        rows = report["pairs"]
        network = report.get("network")
        return await self._persist_run(
            corpus,
            AnalysisRunType.COOCCURRENCE,
            user_id=user_id,
            parameters={
                "unit_type": unit_type,
                "window_size": window_size,
                "top_n": top_n,
                "association_method": report["association_method"],
                "directional": report["directional"],
                "min_frequency": min_frequency,
                "min_count": min_count,
                "include_network": include_network,
                "filters": filters,
                **config_params,
            },
            metrics={
                "unit_count": len(units),
                "pairs_returned": len(rows),
                "pairs_tested": report["pairs_tested"],
                "association_method": report["association_method"],
                "window": report["window"],
                "direction": report["direction"],
                "node_count": (network or {}).get("node_count"),
                "edge_count": (network or {}).get("edge_count"),
            },
            results={
                "cooccurrence": rows,
                "report": report,
                "network": network,
            },
        )

    async def similarity(
        self,
        corpus_id: str,
        *,
        user_id: str,
        unit_type: str,
        method: str = "tfidf_cosine",
        mode: str = "pairwise",
        top_k: int | None = 20,
        min_score: float | None = None,
        group_by: str | None = None,
        centroid_target: str = "between_groups",
        query_text: str | None = None,
        query_unit_id: str | None = None,
        embeddings: dict[str, list[float]] | None = None,
        query_embedding: list[float] | None = None,
        preprocessing_profile_id: str | None = None,
        **filters: Any,
    ) -> AnalysisRun:
        """Document-to-document / unit-to-unit / query-to-document / group-centroid
        similarity over text units selected from the corpus.

        ``method='embedding_cosine'`` requires an explicit ``embeddings`` mapping
        supplied on *this* call — embeddings are never computed or cached by this
        service, so rerunning an embedding-based similarity run requires passing
        ``embeddings`` again (they are intentionally not persisted in run parameters).
        """
        from backend.modules.text_research.infrastructure import similarity as sim_mod

        try:
            canonical_method = sim_mod.normalize_similarity_method(method)
            canonical_mode = sim_mod.normalize_similarity_mode(mode)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        corpus, units, documents = await self._select(
            corpus_id, user_id=user_id, unit_type=unit_type, filters=filters
        )
        config, config_params = await self._resolve_config(
            preprocessing_profile_id, user_id=user_id
        )

        ids = [u.id for u in units]
        texts = [u.text for u in units]

        group_keys: list[str] | None = None
        if canonical_mode == "group_centroid":
            if not group_by:
                raise HTTPException(
                    status_code=422,
                    detail="mode='group_centroid' requires 'group_by' (a document metadata field)",
                )
            doc_lookup = self._document_lookup(units, documents)
            group_keys = [
                str(getattr(doc_lookup.get(u.corpus_document_id), group_by, None) or "unspecified")
                for u in units
            ]

        resolved_query_text = query_text
        query_id = "query"
        if canonical_mode == "query" and canonical_method != "embedding_cosine":
            if query_unit_id:
                query_unit = next((u for u in units if u.id == query_unit_id), None)
                if query_unit is None:
                    raise HTTPException(
                        status_code=422,
                        detail="query_unit_id must reference one of the selected text units",
                    )
                resolved_query_text = query_unit.text
                query_id = query_unit.id
                keep = [i for i, u in enumerate(units) if u.id != query_unit_id]
                ids = [ids[i] for i in keep]
                texts = [texts[i] for i in keep]
            elif not query_text:
                raise HTTPException(
                    status_code=422,
                    detail="mode='query' requires 'query_text' or 'query_unit_id'",
                )

        cache_key = self._cache_key(
            corpus_id=corpus.id,
            unit_type=unit_type,
            units=units,
            config=config,
            filters=filters,
            mode="tokens",
        )

        try:
            report = await asyncio.to_thread(
                quantitative.similarity_for_texts,
                texts,
                ids,
                config,
                method=canonical_method,
                mode=canonical_mode,
                group_keys=group_keys,
                centroid_target=centroid_target,
                query_text=resolved_query_text,
                query_id=query_id,
                embeddings=embeddings,
                query_embedding=query_embedding,
                top_k=top_k,
                min_score=min_score,
                cache_key=cache_key,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        result_rows = report.get("pairs") or report.get("items") or []
        return await self._persist_run(
            corpus,
            AnalysisRunType.SIMILARITY,
            user_id=user_id,
            parameters={
                "unit_type": unit_type,
                "method": canonical_method,
                "mode": canonical_mode,
                "top_k": top_k,
                "min_score": min_score,
                "group_by": group_by,
                "centroid_target": centroid_target,
                "query_text": query_text,
                "query_unit_id": query_unit_id,
                "has_embeddings": bool(embeddings),
                "filters": filters,
                **config_params,
            },
            metrics={
                "method": canonical_method,
                "mode": canonical_mode,
                "item_count": report.get("item_count", len(ids)),
                "rows_returned": len(result_rows),
            },
            results=report,
        )

    async def duplicate_detection(
        self,
        corpus_id: str,
        *,
        user_id: str,
        unit_type: str,
        methods: list[str] | None = None,
        lexical_threshold: float = 0.85,
        char_ngram_size: int = 5,
        use_minhash: bool = False,
        minhash_num_perm: int = 64,
        minhash_shingle_size: int = 3,
        minhash_threshold: float = 0.8,
        max_pairs: int | None = 1000,
        **filters: Any,
    ) -> AnalysisRun:
        """Exact / normalized checksum, lexical near-dup, and optional MinHash
        duplicate detection over text units selected from the corpus.

        This is the same engine ingestion QA uses for its near-duplicate check
        (:mod:`infrastructure.duplicate_detection`), exposed here as an explicit,
        rerunnable analysis over any unit granularity / metadata filter.
        """
        from backend.modules.text_research.infrastructure.duplicate_detection import (
            duplicate_report,
            normalize_duplicate_methods,
        )

        try:
            resolved_methods = normalize_duplicate_methods(methods)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if use_minhash and "minhash" not in resolved_methods:
            resolved_methods.append("minhash")

        corpus, units, _ = await self._select(
            corpus_id, user_id=user_id, unit_type=unit_type, filters=filters
        )
        items = [{"id": u.id, "text": u.text} for u in units]

        try:
            report = await asyncio.to_thread(
                duplicate_report,
                items,
                methods=resolved_methods,
                lexical_threshold=lexical_threshold,
                char_ngram_size=char_ngram_size,
                minhash_num_perm=minhash_num_perm,
                minhash_shingle_size=minhash_shingle_size,
                minhash_threshold=minhash_threshold,
                max_pairs=max_pairs,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return await self._persist_run(
            corpus,
            AnalysisRunType.DUPLICATE_DETECTION,
            user_id=user_id,
            parameters={
                "unit_type": unit_type,
                "methods": resolved_methods,
                "lexical_threshold": lexical_threshold,
                "char_ngram_size": char_ngram_size,
                "use_minhash": use_minhash,
                "minhash_num_perm": minhash_num_perm,
                "minhash_shingle_size": minhash_shingle_size,
                "minhash_threshold": minhash_threshold,
                "max_pairs": max_pairs,
                "filters": filters,
            },
            metrics={"unit_count": len(units), **report["summary"]},
            results=report,
        )

    async def clustering(
        self,
        corpus_id: str,
        *,
        user_id: str,
        unit_type: str,
        n_clusters: int = 5,
        algorithm: str = "kmeans",
        use_svd: bool = False,
        n_svd_components: int = 50,
        top_terms: int = 10,
        random_seed: int = 42,
        preprocessing_profile_id: str | None = None,
        **filters: Any,
    ) -> AnalysisRun:
        from backend.modules.text_research.infrastructure.clustering import run_clustering

        corpus, units, _ = await self._select(
            corpus_id, user_id=user_id, unit_type=unit_type, filters=filters
        )
        config, config_params = await self._resolve_config(preprocessing_profile_id, user_id=user_id)
        try:
            result = await asyncio.to_thread(
                run_clustering,
                [u.text for u in units],
                [u.id for u in units],
                n_clusters=n_clusters,
                algorithm=algorithm,
                config=config if isinstance(config, dict) else config.to_dict(),
                use_svd=use_svd,
                svd_components=n_svd_components,
                random_seed=random_seed,
                top_n_terms=top_terms,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        # Drop non-JSON-serializable matrix handles before persistence.
        persisted = {k: v for k, v in result.items() if k not in {"tfidf_matrix"}}

        return await self._persist_run(
            corpus,
            AnalysisRunType.CLUSTERING,
            user_id=user_id,
            parameters={
                "unit_type": unit_type,
                "n_clusters": n_clusters,
                "algorithm": algorithm,
                "use_svd": use_svd,
                "n_svd_components": n_svd_components,
                "top_terms": top_terms,
                "random_seed": random_seed,
                "filters": filters,
                **config_params,
            },
            metrics={
                "unit_count": len(units),
                "n_clusters": persisted.get("n_clusters"),
                "silhouette_score": persisted.get("silhouette_score"),
            },
            results=persisted,
        )

    async def dimensionality_reduction(
        self,
        corpus_id: str,
        *,
        user_id: str,
        unit_type: str,
        method: str = "svd",
        n_components: int = 2,
        random_seed: int = 42,
        preprocessing_profile_id: str | None = None,
        **filters: Any,
    ) -> AnalysisRun:
        from backend.modules.text_research.infrastructure.clustering import build_tfidf_matrix
        from backend.modules.text_research.infrastructure.dimensionality import reduce_dimensions

        corpus, units, _ = await self._select(
            corpus_id, user_id=user_id, unit_type=unit_type, filters=filters
        )
        config, config_params = await self._resolve_config(preprocessing_profile_id, user_id=user_id)
        cfg = config if isinstance(config, dict) else config.to_dict()

        def _run() -> dict[str, Any]:
            matrix, _ = build_tfidf_matrix([u.text for u in units], cfg)
            return reduce_dimensions(
                matrix,
                [u.id for u in units],
                method=method,
                n_components=n_components,
                random_seed=random_seed,
            )

        try:
            result = await asyncio.to_thread(_run)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return await self._persist_run(
            corpus,
            AnalysisRunType.DIMENSIONALITY_REDUCTION,
            user_id=user_id,
            parameters={
                "unit_type": unit_type,
                "method": method,
                "n_components": n_components,
                "random_seed": random_seed,
                "filters": filters,
                **config_params,
            },
            metrics={"unit_count": len(units), "n_components": n_components},
            results=result,
        )

    async def readability(
        self,
        corpus_id: str,
        *,
        user_id: str,
        unit_type: str,
        **filters: Any,
    ) -> AnalysisRun:
        from backend.modules.text_research.infrastructure.readability import readability_for_units

        corpus, units, _ = await self._select(
            corpus_id, user_id=user_id, unit_type=unit_type, filters=filters
        )
        result = await asyncio.to_thread(
            readability_for_units,
            [u.text for u in units],
            [u.id for u in units],
        )
        return await self._persist_run(
            corpus,
            AnalysisRunType.READABILITY,
            user_id=user_id,
            parameters={"unit_type": unit_type, "filters": filters},
            metrics={
                "unit_count": len(units),
                "flesch_reading_ease": result["corpus"]["flesch_reading_ease"],
                "flesch_kincaid_grade": result["corpus"]["flesch_kincaid_grade"],
            },
            results=result,
        )
