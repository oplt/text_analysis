"""Similarity and near-duplicate quantitative analyses."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import HTTPException

from backend.modules.text_research.application.analysis_executor import (
    estimate_workload,
    should_enqueue_cpu_job,
)
from backend.modules.text_research.application.analysis_operators import invoke_operator
from backend.modules.text_research.application.quantitative_support import (
    QUANT_OP_KEY,
    _content_checksum,
    _prepare_with_identity,
    _resolve_group_keys,
    _resolve_group_value,
    _with_frozen_preprocessing,
)
from backend.modules.text_research.domain.enums import AnalysisRunType
from backend.modules.text_research.domain.models import AnalysisRun
from backend.modules.text_research.infrastructure.preprocessing import PreprocessingConfig

_UNSET = object()


class SimilarityAnalysisMixin:
    """Similarity and near-duplicate quantitative analyses."""

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
        embedding_artifact_id: str | None = None,
        preprocessing_profile_id: str | None = None,
        preprocessing_config: dict[str, Any] | None = None,
        force_inline: bool = False,
        run_async: bool = False,
        existing_run_id: str | None = None,
        **filters: Any,
    ) -> AnalysisRun:
        """Document-to-document / unit-to-unit / query-to-document / group-centroid
        similarity over text units selected from the corpus.

        ``method='embedding_cosine'`` requires either inline ``embeddings`` (forced
        inline only — never queued) or a managed ``embedding_artifact_id`` that the
        worker can reload. Raw vectors are never written to ``parameters_json``.
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

        ids = [u.id for u in units]
        texts = [u.text for u in units]

        n_groups = 0
        if canonical_mode == "group_centroid" and group_by:
            # Cheap cardinality hint for workload (full keys resolved later).
            n_groups = len({_resolve_group_value(d, group_by) for d in documents})
        estimate = estimate_workload(
            analysis_type="similarity",
            n_units=len(units),
            texts=texts,
            pair_mode=canonical_mode,
            requested_top_k=top_k or 0,
            n_groups=n_groups,
        )
        would_enqueue = should_enqueue_cpu_job(
            estimate,
            force_inline=force_inline or bool(existing_run_id),
            force_async=run_async,
        )
        # LATEST-006: never queue embedding_cosine without a managed artifact.
        # Workload auto-enqueue and run_async both hit this path; raw vectors are
        # never written into AnalysisRun.parameters_json.
        if would_enqueue and canonical_method == "embedding_cosine" and not embedding_artifact_id:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Raw embedding vectors cannot be queued; persist/use a managed "
                    "embedding artifact."
                ),
            )
        if canonical_method == "embedding_cosine" and not (embeddings or embedding_artifact_id):
            raise HTTPException(
                status_code=422,
                detail=(
                    "embedding_cosine requires explicit embeddings or a managed embedding artifact"
                ),
            )
        if would_enqueue:
            return await self._enqueue_quantitative(
                corpus,
                AnalysisRunType.SIMILARITY,
                user_id=user_id,
                operation="similarity",
                parameters=_with_frozen_preprocessing(
                    {
                        "unit_type": unit_type,
                        "preprocessing_profile_id": preprocessing_profile_id,
                        "method": method,
                        "mode": mode,
                        "top_k": top_k,
                        "min_score": min_score,
                        "group_by": group_by,
                        "centroid_target": centroid_target,
                        "query_text": query_text,
                        "query_unit_id": query_unit_id,
                        "embedding_artifact_id": embedding_artifact_id,
                        "filters": filters,
                    },
                    preprocessing_config,
                ),
                estimate=estimate,
            )
        existing_run = await self._resolve_existing_run(existing_run_id)
        artifact_metadata: dict[str, Any] | None = None
        resolved_embedding_identity: dict[str, Any] | None = None
        if embedding_artifact_id:
            from backend.modules.text_research.infrastructure.embeddings import (
                embedding_identity_from_artifact_metadata,
                load_managed_embedding_artifact,
            )

            try:
                embeddings, artifact_metadata = load_managed_embedding_artifact(
                    embedding_artifact_id,
                    project_id=corpus.project_id,
                    corpus_id=corpus.id,
                    unit_ids=ids,
                    expected_dim=len(query_embedding) if query_embedding else None,
                )
            except PermissionError as exc:
                raise HTTPException(status_code=403, detail=str(exc)) from exc
            except (KeyError, ValueError, FileNotFoundError) as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            stored = artifact_metadata.get("embedding_identity")
            resolved_embedding_identity = (
                stored
                if isinstance(stored, dict)
                else embedding_identity_from_artifact_metadata(artifact_metadata)
            )
        config, config_params = await self._resolve_config(
            preprocessing_profile_id,
            user_id=user_id,
            preprocessing_config=preprocessing_config,
        )
        prepared, identity = await _prepare_with_identity(
            corpus_id=corpus.id,
            analysis_type="similarity",
            texts=texts,
            config=config,
            units=units,
            unit_type=unit_type,
            filters=filters,
            preprocessing_profile_id=preprocessing_profile_id,
            analysis_parameters={
                "method": canonical_method,
                "mode": canonical_mode,
                "group_by": group_by,
                "top_k": top_k,
                "min_score": min_score,
                "centroid_target": centroid_target,
                "query_text": query_text,
                "query_unit_id": query_unit_id,
                "query_embedding_checksum": _content_checksum(query_embedding)
                if query_embedding
                else None,
                "embedding_checksum": _content_checksum(embeddings) if embeddings else None,
                "embedding_artifact_id": embedding_artifact_id,
                "embedding_artifact_checksum": (
                    artifact_metadata.get("content_checksum") if artifact_metadata else None
                ),
                "embedding_identity": resolved_embedding_identity,
                "has_embeddings": bool(embeddings),
            },
        )
        reused = await self._reuse_deterministic_run(
            corpus,
            user_id=user_id,
            identity={
                QUANT_OP_KEY: "similarity",
                "unit_type": unit_type,
                "method": canonical_method,
                "mode": canonical_mode,
                "top_k": top_k,
                "min_score": min_score,
                "centroid_target": centroid_target,
                "query_text": query_text,
                "query_unit_id": query_unit_id,
                "embedding_artifact_id": embedding_artifact_id,
                "embedding_identity": resolved_embedding_identity,
                "has_embeddings": bool(embeddings) and not embedding_artifact_id,
                "filters": filters,
                **identity,
                **config_params,
            },
            existing_run_id=existing_run_id,
        )
        if reused is not None:
            return reused
        group_keys: list[str] | None = None
        if canonical_mode == "group_centroid":
            if not group_by:
                raise HTTPException(
                    status_code=422,
                    detail="mode='group_centroid' requires 'group_by' (a document metadata field)",
                )
            group_keys = _resolve_group_keys(units, documents, group_by)

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
            elif not query_text:
                raise HTTPException(
                    status_code=422,
                    detail="mode='query' requires 'query_text' or 'query_unit_id'",
                )

        try:
            report = await asyncio.to_thread(
                invoke_operator,
                "similarity",
                prepared,
                {
                    "method": canonical_method,
                    "mode": canonical_mode,
                    "top_k": top_k,
                    "min_score": min_score,
                    "centroid_target": centroid_target,
                    "query_text": resolved_query_text,
                    "query_id": query_id,
                    "embeddings": embeddings,
                    "query_embedding": query_embedding,
                    "exclude_unit_id": (
                        query_unit_id if canonical_mode == "query" and query_unit_id else None
                    ),
                },
                group_keys=group_keys,
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
                "embedding_artifact_id": embedding_artifact_id,
                "centroid_target": centroid_target,
                "query_text": query_text,
                "query_unit_id": query_unit_id,
                "has_embeddings": bool(embeddings),
                "filters": filters,
                **identity,
                **config_params,
            },
            metrics={
                "method": canonical_method,
                "mode": canonical_mode,
                "item_count": report.get("item_count", len(ids)),
                "rows_returned": len(result_rows),
            },
            results={
                **report,
                "corpus_checksum": prepared.corpus_checksum,
                "pipeline_checksum": prepared.pipeline_checksum,
            },
            existing_run=existing_run,
        )

    async def duplicate_detection(
        self,
        corpus_id: str,
        *,
        user_id: str,
        unit_type: str,
        methods: list[str] | None = None,
        lexical_threshold: float | None = None,
        char_ngram_size: int | None = None,
        use_minhash: bool | None = None,
        minhash_num_perm: int | None = None,
        minhash_shingle_size: int | None = None,
        minhash_threshold: float | None = None,
        max_pairs: int | None | object = _UNSET,
        force_inline: bool = False,
        run_async: bool = False,
        existing_run_id: str | None = None,
        **filters: Any,
    ) -> AnalysisRun:
        """Exact / normalized checksum, lexical near-dup, and optional MinHash
        duplicate detection over text units selected from the corpus.

        This is the same engine ingestion QA uses for its near-duplicate check
        (:mod:`infrastructure.duplicate_detection`), exposed here as an explicit,
        rerunnable analysis over any unit granularity / metadata filter.

        Analysis knobs default via :class:`DuplicateDetectionConfig` when omitted.
        """
        from backend.modules.text_research.domain.quantitative_configs import (
            DuplicateDetectionConfig,
        )
        from backend.modules.text_research.infrastructure.duplicate_detection import (
            normalize_duplicate_methods,
        )

        raw_knobs: dict[str, Any] = {"methods": methods}
        if lexical_threshold is not None:
            raw_knobs["lexical_threshold"] = lexical_threshold
        if char_ngram_size is not None:
            raw_knobs["char_ngram_size"] = char_ngram_size
        if use_minhash is not None:
            raw_knobs["use_minhash"] = use_minhash
        if minhash_num_perm is not None:
            raw_knobs["minhash_num_perm"] = minhash_num_perm
        if minhash_shingle_size is not None:
            raw_knobs["minhash_shingle_size"] = minhash_shingle_size
        if minhash_threshold is not None:
            raw_knobs["minhash_threshold"] = minhash_threshold
        if max_pairs is not _UNSET:
            raw_knobs["max_pairs"] = max_pairs
        config = DuplicateDetectionConfig.model_validate(raw_knobs)
        methods = config.methods
        lexical_threshold = config.lexical_threshold
        char_ngram_size = config.char_ngram_size
        use_minhash = config.use_minhash
        minhash_num_perm = config.minhash_num_perm
        minhash_shingle_size = config.minhash_shingle_size
        minhash_threshold = config.minhash_threshold
        max_pairs = config.max_pairs

        try:
            resolved_methods = normalize_duplicate_methods(methods)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if use_minhash and "minhash" not in resolved_methods:
            resolved_methods.append("minhash")

        corpus, units, _ = await self._select(
            corpus_id, user_id=user_id, unit_type=unit_type, filters=filters
        )
        estimate = estimate_workload(
            analysis_type="duplicate_detection",
            n_units=len(units),
            texts=[u.text for u in units],
            pair_mode="duplicate",
        )
        if should_enqueue_cpu_job(
            estimate,
            force_inline=force_inline or bool(existing_run_id),
            force_async=run_async,
        ):
            return await self._enqueue_quantitative(
                corpus,
                AnalysisRunType.DUPLICATE_DETECTION,
                user_id=user_id,
                operation="duplicate_detection",
                parameters={
                    "unit_type": unit_type,
                    "methods": methods,
                    "lexical_threshold": lexical_threshold,
                    "char_ngram_size": char_ngram_size,
                    "use_minhash": use_minhash,
                    "minhash_num_perm": minhash_num_perm,
                    "minhash_shingle_size": minhash_shingle_size,
                    "minhash_threshold": minhash_threshold,
                    "max_pairs": max_pairs,
                    "filters": filters,
                },
                estimate=estimate,
            )
        existing_run = await self._resolve_existing_run(existing_run_id)
        prepared, identity = await _prepare_with_identity(
            corpus_id=corpus.id,
            analysis_type="duplicate_detection",
            texts=[u.text for u in units],
            config=PreprocessingConfig(),
            units=units,
            unit_type=unit_type,
            filters=filters,
            analysis_parameters={
                "methods": resolved_methods,
                "lexical_threshold": lexical_threshold,
                "char_ngram_size": char_ngram_size,
                "use_minhash": use_minhash,
                "minhash_num_perm": minhash_num_perm,
                "minhash_shingle_size": minhash_shingle_size,
                "minhash_threshold": minhash_threshold,
                "max_pairs": max_pairs,
            },
        )
        try:
            report = await asyncio.to_thread(
                invoke_operator,
                "duplicate_detection",
                prepared,
                {
                    "methods": resolved_methods,
                    "lexical_threshold": lexical_threshold,
                    "char_ngram_size": char_ngram_size,
                    "minhash_num_perm": minhash_num_perm,
                    "minhash_shingle_size": minhash_shingle_size,
                    "minhash_threshold": minhash_threshold,
                    "max_pairs": max_pairs,
                },
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
                **identity,
            },
            metrics={"unit_count": len(units), **report["summary"]},
            results={
                **report,
                "corpus_checksum": prepared.corpus_checksum,
                "pipeline_checksum": prepared.pipeline_checksum,
            },
            existing_run=existing_run,
        )
