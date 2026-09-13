"""Matrix / contrastive quantitative analyses: DFM, keyness, co-occurrence."""

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
    _prepare_with_identity,
    _with_frozen_preprocessing,
)
from backend.modules.text_research.domain.analysis_specification import normalize_corpus_filters
from backend.modules.text_research.domain.enums import AnalysisRunType
from backend.modules.text_research.domain.models import AnalysisRun
from backend.modules.text_research.infrastructure import quantitative
from backend.modules.text_research.infrastructure.prepared_corpus_builder import (
    prepare_texts_cached_async,
)


class MatrixAnalysisMixin:
    """Matrix / contrastive quantitative analyses: DFM, keyness, co-occurrence."""

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
        preprocessing_config: dict[str, Any] | None = None,
        force_sparse_only: bool = False,
        trim: dict[str, Any] | None = None,
        force_inline: bool = False,
        run_async: bool = False,
        existing_run_id: str | None = None,
        **filters: Any,
    ) -> AnalysisRun:
        try:
            weighting = quantitative.normalize_dfm_weighting(weighting)
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
            analysis_type="dfm",
            n_units=len(units),
            texts=texts,
        )
        if should_enqueue_cpu_job(
            estimate,
            force_inline=force_inline or bool(existing_run_id),
            force_async=run_async,
        ):
            return await self._enqueue_quantitative(
                corpus,
                AnalysisRunType.DFM,
                user_id=user_id,
                operation="dfm",
                parameters=_with_frozen_preprocessing(
                    {
                        "unit_type": unit_type,
                        "preprocessing_profile_id": preprocessing_profile_id,
                        "weighting": weighting,
                        "k1": k1,
                        "b": b,
                        "smooth_idf": smooth_idf,
                        "force_sparse_only": force_sparse_only,
                        "trim": trim,
                        "filters": filters,
                    },
                    preprocessing_config,
                ),
                estimate=estimate,
            )
        existing_run = await self._resolve_existing_run(existing_run_id)
        prepared, identity = await _prepare_with_identity(
            corpus_id=corpus.id,
            analysis_type="dfm",
            texts=texts,
            config=config,
            units=units,
            unit_type=unit_type,
            filters=filters,
            preprocessing_profile_id=preprocessing_profile_id,
            analysis_parameters={
                "weighting": weighting,
                "trim": trim,
                "k1": k1,
                "b": b,
                "smooth_idf": smooth_idf,
                "force_sparse_only": force_sparse_only,
            },
        )
        build_kwargs: dict[str, Any] = {
            "weighting": weighting,
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
            operator_result = await asyncio.to_thread(
                invoke_operator,
                "dfm",
                prepared,
                build_kwargs,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        result = operator_result["dfm"]
        summary = operator_result["summary"]
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
                **identity,
                **config_params,
            },
            metrics={
                "unit_count": summary["unit_count"],
                "feature_count": summary["feature_count"],
                "density": summary["density"],
                "sparsity": summary.get("sparsity"),
                "nnz": summary["nnz"],
                "storage": summary["storage"],
                "mode": summary["mode"],
                "trim": summary.get("trim"),
                "estimated_memory_bytes": summary.get("estimated_memory_bytes"),
                "scientific_warnings": summary.get("scientific_warnings"),
            },
            results={
                "summary": summary,
                "dfm": persisted,
                "corpus_checksum": prepared.corpus_checksum,
                "pipeline_checksum": prepared.pipeline_checksum,
            },
            existing_run=existing_run,
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
        preprocessing_config: dict[str, Any] | None = None,
        top_n: int = 50,
        force_inline: bool = False,
        run_async: bool = False,
        existing_run_id: str | None = None,
    ) -> AnalysisRun:
        if not filters_a or not filters_b:
            raise HTTPException(
                status_code=400,
                detail=(
                    "filters_a and filters_b are required — choose comparison "
                    "groups from corpus metadata"
                ),
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
        estimate = estimate_workload(
            analysis_type="keyness",
            n_units=len(units_a) + len(units_b),
            texts=[u.text for u in units_a] + [u.text for u in units_b],
            requested_top_k=top_n,
        )
        if should_enqueue_cpu_job(
            estimate,
            force_inline=force_inline or bool(existing_run_id),
            force_async=run_async,
        ):
            return await self._enqueue_quantitative(
                corpus,
                AnalysisRunType.KEYNESS,
                user_id=user_id,
                operation="keyness",
                parameters=_with_frozen_preprocessing(
                    {
                        "unit_type": unit_type,
                        "preprocessing_profile_id": preprocessing_profile_id,
                        "filters_a": filters_a,
                        "filters_b": filters_b,
                        "group_field": group_field,
                        "method": method,
                        "correction": correction,
                        "min_frequency": min_frequency,
                        "top_n": top_n,
                        "filters": {},
                    },
                    preprocessing_config,
                ),
                estimate=estimate,
            )
        existing_run = await self._resolve_existing_run(existing_run_id)
        config, config_params = await self._resolve_config(
            preprocessing_profile_id,
            user_id=user_id,
            preprocessing_config=preprocessing_config,
        )
        prepared_a, identity = await _prepare_with_identity(
            corpus_id=corpus.id,
            analysis_type="keyness",
            texts=[u.text for u in units_a],
            config=config,
            units=units_a,
            unit_type=unit_type,
            filters=filters_a,
            preprocessing_profile_id=preprocessing_profile_id,
            analysis_parameters={
                "method": method,
                "top_n": top_n,
                "min_frequency": min_frequency,
                "correction": correction,
                "group_field": group_field,
                "filters_a": normalize_corpus_filters(filters_a),
                "filters_b": normalize_corpus_filters(filters_b),
            },
        )
        prepared_b = await prepare_texts_cached_async(
            [u.text for u in units_b],
            config.to_dict(),
            corpus_id=corpus.id,
            unit_type=unit_type,
            unit_ids=[u.id for u in units_b],
            document_ids=[u.corpus_document_id for u in units_b],
            filters=filters_b,
            operation_config={},
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
                invoke_operator,
                "keyness",
                prepared_a,
                {
                    "top_n": top_n,
                    "method": method,
                    "min_frequency": min_frequency,
                    "correction": correction,
                    "group_a_label": label_a,
                    "group_b_label": label_b,
                    "group_field": inferred_field,
                },
                prepared_b=prepared_b,
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
                **identity,
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
            results={
                "keyness": rows,
                "report": report,
                "corpus_checksum": prepared_a.corpus_checksum,
                "pipeline_checksum": prepared_a.pipeline_checksum,
            },
            existing_run=existing_run,
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
        preprocessing_config: dict[str, Any] | None = None,
        force_inline: bool = False,
        run_async: bool = False,
        existing_run_id: str | None = None,
        **filters: Any,
    ) -> AnalysisRun:
        corpus, units, _ = await self._select(
            corpus_id, user_id=user_id, unit_type=unit_type, filters=filters
        )
        config, config_params = await self._resolve_config(
            preprocessing_profile_id,
            user_id=user_id,
            preprocessing_config=preprocessing_config,
        )

        estimate = estimate_workload(
            analysis_type="cooccurrence",
            n_units=len(units),
            texts=[u.text for u in units],
            pair_mode="cooccurrence",
            window_size=window_size,
            requested_top_k=top_n,
        )
        if should_enqueue_cpu_job(
            estimate,
            force_inline=force_inline or bool(existing_run_id),
            force_async=run_async,
        ):
            return await self._enqueue_quantitative(
                corpus,
                AnalysisRunType.COOCCURRENCE,
                user_id=user_id,
                operation="cooccurrence",
                parameters=_with_frozen_preprocessing(
                    {
                        "unit_type": unit_type,
                        "preprocessing_profile_id": preprocessing_profile_id,
                        "window_size": window_size,
                        "top_n": top_n,
                        "association_method": association_method,
                        "directional": directional,
                        "min_frequency": min_frequency,
                        "min_count": min_count,
                        "include_network": include_network,
                        "filters": filters,
                    },
                    preprocessing_config,
                ),
                estimate=estimate,
            )
        existing_run = await self._resolve_existing_run(existing_run_id)
        prepared, identity = await _prepare_with_identity(
            corpus_id=corpus.id,
            analysis_type="cooccurrence",
            texts=[u.text for u in units],
            config=config,
            units=units,
            unit_type=unit_type,
            filters=filters,
            preprocessing_profile_id=preprocessing_profile_id,
            analysis_parameters={
                "window_size": window_size,
                "top_n": top_n,
                "association_method": association_method,
                "directional": directional,
                "min_frequency": min_frequency,
                "min_count": min_count,
                "include_network": include_network,
            },
        )
        try:
            report = await asyncio.to_thread(
                invoke_operator,
                "cooccurrence",
                prepared,
                {
                    "window_size": window_size,
                    "top_n": top_n,
                    "association_method": association_method,
                    "directional": directional,
                    "min_frequency": min_frequency,
                    "min_count": min_count,
                    "include_network": include_network,
                },
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
                **identity,
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
                "corpus_checksum": prepared.corpus_checksum,
                "pipeline_checksum": prepared.pipeline_checksum,
            },
            existing_run=existing_run,
        )
