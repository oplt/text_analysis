"""Corpus statistics, frequencies, n-grams, KWIC, dictionary, and readability."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import HTTPException

from backend.modules.text_research.application.analysis_executor import (
    estimate_workload,
    execute_or_enqueue,
    run_cpu_bound,
    should_enqueue_cpu_job,
)
from backend.modules.text_research.application.analysis_operators import invoke_operator
from backend.modules.text_research.application.quantitative_support import (
    QUANT_OP_KEY,
    _content_checksum,
    _prepare_with_identity,
    _resolve_group_keys,
    _with_frozen_preprocessing,
)
from backend.modules.text_research.domain.enums import AnalysisRunType
from backend.modules.text_research.domain.models import AnalysisRun
from backend.modules.text_research.infrastructure import quantitative
from backend.modules.text_research.infrastructure.preprocessing import PreprocessingConfig


class LexicalAnalysisMixin:
    """Corpus statistics, frequencies, n-grams, KWIC, dictionary, and readability."""

    async def corpus_stats(
        self,
        corpus_id: str,
        *,
        user_id: str,
        unit_type: str,
        preprocessing_profile_id: str | None = None,
        preprocessing_config: dict[str, Any] | None = None,
        group_by: list[str] | None = None,
        **filters: Any,
    ) -> AnalysisRun:
        corpus, units, documents = await self._select(
            corpus_id, user_id=user_id, unit_type=unit_type, filters=filters
        )
        config, config_params = await self._resolve_config(
            preprocessing_profile_id,
            user_id=user_id,
            preprocessing_config=preprocessing_config,
        )
        texts = [u.text for u in units]

        doc_lookup = self._document_lookup(units, documents)

        def group_counts(field: str) -> dict[str, int]:
            counts: dict[str, int] = {}
            for unit in units:
                doc = doc_lookup.get(unit.corpus_document_id)
                if doc is None:
                    key = "unspecified"
                elif hasattr(doc, "get_field_value"):
                    value = doc.get_field_value(field)
                    key = str(value) if value is not None and value != "" else "unspecified"
                else:
                    key = str(getattr(doc, field, None) or "unspecified")
                counts[key] = counts.get(key, 0) + 1
            return counts

        # Generic metadata slicing only — no hardcoded research dimensions.
        fields = [f for f in (group_by or []) if f]
        breakdowns = {field: group_counts(field) for field in fields}
        prepared, identity = await _prepare_with_identity(
            corpus_id=corpus.id,
            analysis_type="corpus_stats",
            texts=texts,
            config=config,
            units=units,
            unit_type=unit_type,
            filters=filters,
            preprocessing_profile_id=preprocessing_profile_id,
            analysis_parameters={"group_by": fields},
        )
        stats = await asyncio.to_thread(invoke_operator, "corpus_stats", prepared, {})
        return await self._persist_run(
            corpus,
            AnalysisRunType.CORPUS_STATS,
            user_id=user_id,
            parameters={
                "unit_type": unit_type,
                "filters": filters,
                "group_by": fields,
                "pipeline_workflow": list(quantitative.PIPELINE_STAGES),
                **identity,
                **config_params,
            },
            metrics=stats,
            results={
                "breakdowns": breakdowns,
                "lexical_diversity": stats.get("lexical_diversity"),
                "pipeline": stats.get("pipeline"),
                "corpus_checksum": prepared.corpus_checksum,
                "pipeline_checksum": prepared.pipeline_checksum,
            },
        )

    async def frequencies(
        self,
        corpus_id: str,
        *,
        user_id: str,
        unit_type: str,
        preprocessing_profile_id: str | None = None,
        preprocessing_config: dict[str, Any] | None = None,
        top_n: int = 50,
        rate_per: float = 1000,
        group_by: str | None = None,
        force_inline: bool = False,
        run_async: bool = False,
        existing_run_id: str | None = None,
        **filters: Any,
    ) -> AnalysisRun:
        corpus, units, documents = await self._select(
            corpus_id, user_id=user_id, unit_type=unit_type, filters=filters
        )
        texts = [u.text for u in units]
        estimate = estimate_workload(
            analysis_type="frequencies",
            n_units=len(units),
            n_documents=len(documents),
            texts=texts,
            requested_top_k=top_n,
        )
        request_params = _with_frozen_preprocessing(
            {
                "unit_type": unit_type,
                "preprocessing_profile_id": preprocessing_profile_id,
                "top_n": top_n,
                "rate_per": rate_per,
                "group_by": group_by,
                "filters": filters,
            },
            preprocessing_config,
        )

        async def _enqueue():
            return await self._enqueue_quantitative(
                corpus,
                AnalysisRunType.FREQUENCY_ANALYSIS,
                user_id=user_id,
                operation="frequencies",
                parameters=request_params,
                estimate=estimate,
            )

        async def _inline():
            existing_run = await self._resolve_existing_run(existing_run_id)
            config, config_params = await self._resolve_config(
                preprocessing_profile_id,
                user_id=user_id,
                preprocessing_config=preprocessing_config,
            )
            prepared, identity = await _prepare_with_identity(
                corpus_id=corpus.id,
                analysis_type="frequencies",
                texts=texts,
                config=config,
                units=units,
                unit_type=unit_type,
                filters=filters,
                preprocessing_profile_id=preprocessing_profile_id,
                analysis_parameters={"top_n": top_n, "rate_per": rate_per, "group_by": group_by},
            )
            reused = await self._reuse_deterministic_run(
                corpus,
                user_id=user_id,
                identity={
                    QUANT_OP_KEY: "frequencies",
                    **request_params,
                    **identity,
                    **config_params,
                },
                existing_run_id=existing_run_id,
            )
            if reused is not None:
                return reused
            group_keys = _resolve_group_keys(units, documents, group_by)

            report = await run_cpu_bound(
                invoke_operator,
                "frequencies",
                prepared,
                {"top_n": top_n, "rate_per": rate_per, "group_by": group_by},
                group_keys=group_keys,
            )
            rows = report["frequencies"]
            metadata = report["metadata"]
            return await self._persist_run(
                corpus,
                AnalysisRunType.FREQUENCY_ANALYSIS,
                user_id=user_id,
                parameters={
                    **request_params,
                    **identity,
                    **config_params,
                },
                metrics={
                    "unit_count": metadata["unit_count"],
                    "token_count": metadata["token_count"],
                    "vocabulary_size": metadata["vocabulary_size"],
                    "unique_terms_returned": metadata["terms_returned"],
                    "rate_per": metadata["rate_per"],
                },
                results={
                    "frequencies": rows,
                    "metadata": metadata,
                    "corpus_checksum": prepared.corpus_checksum,
                    "pipeline_checksum": prepared.pipeline_checksum,
                },
                existing_run=existing_run,
            )

        return await execute_or_enqueue(
            estimate=estimate,
            inline=_inline,
            enqueue=_enqueue,
            force_inline=force_inline or bool(existing_run_id),
            force_async=run_async,
        )

    async def ngrams(
        self,
        corpus_id: str,
        *,
        user_id: str,
        unit_type: str,
        n: int = 2,
        preprocessing_profile_id: str | None = None,
        preprocessing_config: dict[str, Any] | None = None,
        top_n: int = 50,
        rate_per: float = 1000,
        skip: int = 0,
        force_inline: bool = False,
        run_async: bool = False,
        existing_run_id: str | None = None,
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
        config, config_params = await self._resolve_config(
            preprocessing_profile_id,
            user_id=user_id,
            preprocessing_config=preprocessing_config,
        )
        texts = [u.text for u in units]

        estimate = estimate_workload(
            analysis_type="ngrams",
            n_units=len(units),
            texts=texts,
            requested_top_k=top_n,
        )
        if should_enqueue_cpu_job(
            estimate,
            force_inline=force_inline or bool(existing_run_id),
            force_async=run_async,
        ):
            return await self._enqueue_quantitative(
                corpus,
                AnalysisRunType.NGRAM_ANALYSIS,
                user_id=user_id,
                operation="ngrams",
                parameters=_with_frozen_preprocessing(
                    {
                        "unit_type": unit_type,
                        "preprocessing_profile_id": preprocessing_profile_id,
                        "n": n,
                        "top_n": top_n,
                        "rate_per": rate_per,
                        "skip": skip,
                        "filters": filters,
                    },
                    preprocessing_config,
                ),
                estimate=estimate,
            )
        existing_run = await self._resolve_existing_run(existing_run_id)
        prepared, identity = await _prepare_with_identity(
            corpus_id=corpus.id,
            analysis_type="ngrams",
            texts=texts,
            config=config,
            units=units,
            unit_type=unit_type,
            filters=filters,
            preprocessing_profile_id=preprocessing_profile_id,
            analysis_parameters={"n": n, "top_n": top_n, "rate_per": rate_per, "skip": skip},
        )
        report = await asyncio.to_thread(
            invoke_operator,
            "ngrams",
            prepared,
            {"n": n, "top_n": top_n, "rate_per": rate_per, "skip": skip},
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
                **identity,
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
            results={
                "ngrams": rows,
                "metadata": metadata,
                "corpus_checksum": prepared.corpus_checksum,
                "pipeline_checksum": prepared.pipeline_checksum,
            },
            existing_run=existing_run,
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
        query_language: str | None = None,
        token_attribute: str | None = None,
        max_matches: int | None = None,
        preprocessing_profile_id: str | None = None,
        preprocessing_config: dict[str, Any] | None = None,
        **filters: Any,
    ) -> AnalysisRun:
        corpus, units, documents = await self._select(
            corpus_id, user_id=user_id, unit_type=unit_type, filters=filters
        )
        config, config_params = await self._resolve_config(
            preprocessing_profile_id,
            user_id=user_id,
            preprocessing_config=preprocessing_config,
        )
        prepared, identity = await _prepare_with_identity(
            corpus_id=corpus.id,
            analysis_type="kwic",
            texts=[u.text for u in units],
            config=config,
            units=units,
            unit_type=unit_type,
            filters=filters,
            preprocessing_profile_id=preprocessing_profile_id,
            analysis_parameters={
                "keyword": keyword,
                "window_size": window_size,
                "case_sensitive": case_sensitive,
                "query_mode": query_mode,
                "query_language": query_language,
                "token_attribute": token_attribute,
                "max_matches": max_matches,
            },
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
            operator_result = await asyncio.to_thread(
                invoke_operator,
                "kwic",
                prepared,
                {
                    "keyword": keyword,
                    "window_size": window_size,
                    "case_sensitive": case_sensitive,
                    "query_mode": query_mode,
                    "query_language": query_language,
                    "token_attribute": token_attribute,
                    "max_matches": max_matches,
                },
                unit_metadata=payload,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        matches = operator_result["matches"]
        from backend.modules.text_research.application.analysis_policy import (
            validate_analysis_policy,
        )

        prep_raw = getattr(prepared, "preprocessing_profile", None)
        prep_dict = prep_raw if isinstance(prep_raw, dict) else None
        policy_warnings = validate_analysis_policy(
            analysis_type="kwic",
            case_sensitive=case_sensitive,
            preprocessing=prep_dict,
            language=query_language,
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
                "query_mode": query_mode,
                "query_language": query_language,
                "token_attribute": token_attribute,
                "max_matches": max_matches,
                "filters": filters,
                **identity,
                **config_params,
            },
            metrics={"unit_count": len(units), "match_count": len(matches)},
            results={
                "matches": matches,
                "scientific_warnings": policy_warnings,
                "corpus_checksum": prepared.corpus_checksum,
                "pipeline_checksum": prepared.pipeline_checksum,
            },
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
        preprocessing_config: dict[str, Any] | None = None,
        run_async: bool = False,
        frozen_dictionary_spec: dict[str, Any] | None = None,
        **filters: Any,
    ) -> AnalysisRun:
        if run_async:
            raise HTTPException(
                status_code=422,
                detail="dictionary does not support asynchronous execution.",
            )
        from backend.modules.text_research.application.dictionary_service import DictionaryService
        from backend.modules.text_research.infrastructure.dictionary_matcher import (
            parse_dictionary_payload,
        )

        corpus, units, documents = await self._select(
            corpus_id, user_id=user_id, unit_type=unit_type, filters=filters
        )
        config, config_params = await self._resolve_config(
            preprocessing_profile_id,
            user_id=user_id,
            preprocessing_config=preprocessing_config,
        )
        texts = [u.text for u in units]
        doc_lookup = self._document_lookup(units, documents)

        dictionary_meta: dict[str, Any] = {"source": "user"}
        # Precedence (TASK-016 / FE contract): when custom ``dictionary_terms`` are
        # supplied together with ``dictionary_id``, the custom terms override the
        # stored dictionary leaf expressions while retaining dictionary identity
        # metadata. Explicit ``hierarchy`` always wins over flat terms.
        # Exact reproduce pins ``frozen_dictionary_spec`` so live dictionary edits
        # cannot change scientific inputs.
        if frozen_dictionary_spec is not None:
            spec = parse_dictionary_payload(
                frozen_dictionary_spec,
                language=dictionary_language or frozen_dictionary_spec.get("language"),
            )
            dictionary_meta["terms_source"] = "frozen_dictionary_spec"
            if frozen_dictionary_spec.get("name"):
                dictionary_meta["name"] = frozen_dictionary_spec.get("name")
            if frozen_dictionary_spec.get("version"):
                dictionary_meta["version"] = frozen_dictionary_spec.get("version")
            if frozen_dictionary_spec.get("description"):
                dictionary_meta["description"] = frozen_dictionary_spec.get("description")
        elif dictionary_id:
            dictionary = await DictionaryService(self.db).get_dictionary(
                dictionary_id, user_id=user_id
            )
            if dictionary.project_id != corpus.project_id:
                raise HTTPException(
                    status_code=400,
                    detail="Dictionary does not belong to this corpus project",
                )
            base_spec = DictionaryService.get_spec(dictionary)
            resolved_language = dictionary_language or base_spec.language
            if hierarchy is not None:
                spec = parse_dictionary_payload(
                    {
                        "hierarchy": hierarchy,
                        "exclusions": exclusions if exclusions is not None else [],
                        "language": resolved_language,
                        "source": "user",
                    },
                    language=resolved_language,
                )
                dictionary_meta["terms_source"] = "request_hierarchy_override"
            elif dictionary_terms:
                spec = parse_dictionary_payload(
                    {
                        "terms": dictionary_terms,
                        "exclusions": exclusions
                        if exclusions is not None
                        else [e.to_dict() for e in base_spec.exclusions],
                        "language": resolved_language,
                        "source": "user",
                    },
                    language=resolved_language,
                )
                dictionary_meta["terms_source"] = "request_terms_override"
            else:
                spec = base_spec
                if dictionary_language:
                    spec.language = dictionary_language
                dictionary_meta["terms_source"] = "dictionary_id"
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

        # Build scientific identity only after resolving stored dictionary content
        # and request overrides (checksum of normalized resolved dict).
        prepared, identity = await _prepare_with_identity(
            corpus_id=corpus.id,
            analysis_type="dictionary",
            texts=texts,
            config=config,
            units=units,
            unit_type=unit_type,
            filters=filters,
            preprocessing_profile_id=preprocessing_profile_id,
            analysis_parameters={
                "dictionary_id": dictionary_id,
                "dictionary_version": dictionary_meta.get("version"),
                "dictionary_content_checksum": _content_checksum(spec.to_dict()),
                "exclusions": [entry.to_dict() for entry in spec.exclusions],
                "language": spec.language,
                "case_sensitive": case_sensitive,
                "rate_per": rate_per,
                "group_by": group_by,
                "terms": spec.to_dict(),
                "hierarchy": hierarchy,
            },
        )
        group_keys = _resolve_group_keys(units, documents, group_by)

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
                invoke_operator,
                "dictionary",
                prepared,
                {
                    "case_sensitive": case_sensitive,
                    "rate_per": rate_per,
                    "unit_ids": [u.id for u in units],
                    "metadata": unit_metadata,
                    "dictionary_metadata": dictionary_meta,
                },
                group_keys=group_keys,
                dictionary_spec=spec,
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
                **identity,
                **config_params,
            },
            metrics={
                "total_hits": result["total_hits"],
                "normalized_hits": result["normalized_hits"],
                "hits_per_1000_tokens": result["hits_per_1000_tokens"],
                "document_prevalence": result["document_prevalence"],
                "match_count": len(result.get("matches") or []),
            },
            results={
                **result,
                "corpus_checksum": prepared.corpus_checksum,
                "pipeline_checksum": prepared.pipeline_checksum,
            },
        )

    async def readability(
        self,
        corpus_id: str,
        *,
        user_id: str,
        unit_type: str,
        **filters: Any,
    ) -> AnalysisRun:
        from backend.modules.text_research.application.analysis_policy import (
            validate_analysis_policy,
        )

        corpus, units, _ = await self._select(
            corpus_id, user_id=user_id, unit_type=unit_type, filters=filters
        )
        config = PreprocessingConfig()
        language = filters.get("language") or getattr(config, "language", None)
        policy_warnings = validate_analysis_policy(
            analysis_type="readability",
            language=language if isinstance(language, str) else None,
            preprocessing=config.to_dict(),
        )
        prepared, identity = await _prepare_with_identity(
            corpus_id=corpus.id,
            analysis_type="readability",
            texts=[u.text for u in units],
            config=config,
            units=units,
            unit_type=unit_type,
            filters=filters,
            preprocessing_profile_id=None,
        )
        result = await asyncio.to_thread(invoke_operator, "readability", prepared, {})
        if policy_warnings:
            result = {
                **result,
                "scientific_warnings": [
                    *list(result.get("scientific_warnings") or []),
                    *policy_warnings,
                ],
            }
        return await self._persist_run(
            corpus,
            AnalysisRunType.READABILITY,
            user_id=user_id,
            parameters={"unit_type": unit_type, "filters": filters, **identity},
            metrics={
                "unit_count": len(units),
                "flesch_reading_ease": result["corpus"]["flesch_reading_ease"],
                "flesch_kincaid_grade": result["corpus"]["flesch_kincaid_grade"],
            },
            results={
                **result,
                "corpus_checksum": prepared.corpus_checksum,
                "pipeline_checksum": prepared.pipeline_checksum,
            },
        )
