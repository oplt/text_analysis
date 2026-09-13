"""Exploratory quantitative analyses: clustering and dimensionality reduction."""

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
from backend.modules.text_research.domain.enums import AnalysisRunType
from backend.modules.text_research.domain.models import AnalysisRun


class ExploratoryAnalysisMixin:
    """Exploratory quantitative analyses: clustering and dimensionality reduction."""

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
        preprocessing_config: dict[str, Any] | None = None,
        force_inline: bool = False,
        run_async: bool = False,
        existing_run_id: str | None = None,
        **filters: Any,
    ) -> AnalysisRun:
        corpus, units, _ = await self._select(
            corpus_id, user_id=user_id, unit_type=unit_type, filters=filters
        )
        if n_clusters >= max(len(units), 1):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"n_clusters ({n_clusters}) must be less than the number of "
                    f"selected observations ({len(units)})"
                ),
            )
        config, config_params = await self._resolve_config(
            preprocessing_profile_id,
            user_id=user_id,
            preprocessing_config=preprocessing_config,
        )

        estimate = estimate_workload(
            analysis_type="clustering",
            n_units=len(units),
            texts=[u.text for u in units],
            n_clusters=n_clusters,
        )
        if should_enqueue_cpu_job(
            estimate, force_inline=force_inline or bool(existing_run_id), force_async=run_async
        ):
            return await self._enqueue_quantitative(
                corpus,
                AnalysisRunType.CLUSTERING,
                user_id=user_id,
                operation="clustering",
                parameters=_with_frozen_preprocessing(
                    {
                        "unit_type": unit_type,
                        "preprocessing_profile_id": preprocessing_profile_id,
                        "n_clusters": n_clusters,
                        "algorithm": algorithm,
                        "use_svd": use_svd,
                        "n_svd_components": n_svd_components,
                        "top_terms": top_terms,
                        "random_seed": random_seed,
                        "filters": filters,
                    },
                    preprocessing_config,
                ),
                estimate=estimate,
            )
        existing_run = await self._resolve_existing_run(existing_run_id)
        prepared, identity = await _prepare_with_identity(
            corpus_id=corpus.id,
            analysis_type="clustering",
            texts=[u.text for u in units],
            config=config,
            units=units,
            unit_type=unit_type,
            filters=filters,
            preprocessing_profile_id=preprocessing_profile_id,
            random_seed=random_seed,
            analysis_parameters={
                "n_clusters": n_clusters,
                "algorithm": algorithm,
                "use_svd": use_svd,
                "n_svd_components": n_svd_components,
                "top_terms": top_terms,
                "random_seed": random_seed,
            },
        )
        try:
            persisted = await asyncio.to_thread(
                invoke_operator,
                "clustering",
                prepared,
                {
                    "n_clusters": n_clusters,
                    "algorithm": algorithm,
                    "use_svd": use_svd,
                    "n_svd_components": n_svd_components,
                    "top_terms": top_terms,
                    "random_seed": random_seed,
                },
                random_seed=random_seed,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

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
                **identity,
                **config_params,
            },
            metrics={
                "unit_count": len(units),
                "n_clusters": persisted.get("n_clusters"),
                "silhouette_score": persisted.get("silhouette_score"),
            },
            results={
                **persisted,
                "corpus_checksum": prepared.corpus_checksum,
                "pipeline_checksum": prepared.pipeline_checksum,
            },
            existing_run=existing_run,
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
            analysis_type="dimensionality_reduction",
            n_units=len(units),
            texts=[u.text for u in units],
        )
        if should_enqueue_cpu_job(
            estimate,
            force_inline=force_inline or bool(existing_run_id),
            force_async=run_async,
        ):
            return await self._enqueue_quantitative(
                corpus,
                AnalysisRunType.DIMENSIONALITY_REDUCTION,
                user_id=user_id,
                operation="dimensionality_reduction",
                parameters=_with_frozen_preprocessing(
                    {
                        "unit_type": unit_type,
                        "preprocessing_profile_id": preprocessing_profile_id,
                        "method": method,
                        "n_components": n_components,
                        "random_seed": random_seed,
                        "filters": filters,
                    },
                    preprocessing_config,
                ),
                estimate=estimate,
            )
        existing_run = await self._resolve_existing_run(existing_run_id)
        prepared, identity = await _prepare_with_identity(
            corpus_id=corpus.id,
            analysis_type="dimensionality_reduction",
            texts=[u.text for u in units],
            config=config,
            units=units,
            unit_type=unit_type,
            filters=filters,
            preprocessing_profile_id=preprocessing_profile_id,
            random_seed=random_seed,
            analysis_parameters={
                "method": method,
                "n_components": n_components,
                "random_seed": random_seed,
            },
        )
        try:
            result = await asyncio.to_thread(
                invoke_operator,
                "dimensionality_reduction",
                prepared,
                {
                    "method": method,
                    "n_components": n_components,
                    "random_seed": random_seed,
                },
                random_seed=random_seed,
            )
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
                **identity,
                **config_params,
            },
            metrics={"unit_count": len(units), "n_components": n_components},
            results=result,
            existing_run=existing_run,
        )
