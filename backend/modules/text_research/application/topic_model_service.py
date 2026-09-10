"""Topic modeling (LDA/NMF) training, persistence, and human topic naming.

Also exposes K-sweep (§42, systematic model-size comparison) and multi-seed
stability (§40) as persisted, auditable ``AnalysisRun`` records — same
auditability guarantee as full training runs, without requiring a training
run per comparison point to be individually inspected.
"""

from __future__ import annotations

import random
from collections import Counter
from datetime import UTC, datetime
from typing import Any

from fastapi import HTTPException

from backend.modules.text_research.application.access import ResearchAccessMixin
from backend.modules.text_research.application.analysis_executor import (
    attach_run_identity,
    build_spec_from_request,
)
from backend.modules.text_research.application.quantitative_analysis_service import (
    _filter_kwargs,
)
from backend.modules.text_research.domain.enums import AnalysisRunStatus, AnalysisRunType
from backend.modules.text_research.domain.models import AnalysisRun, TopicLabel, dumps, loads
from backend.modules.text_research.domain.prepared_corpus import (
    PreparedCorpusArtifact,
    build_prepared_artifact,
)
from backend.modules.text_research.infrastructure import model_storage
from backend.modules.text_research.infrastructure.prepared_corpus_builder import (
    prepare_texts,
    prepare_texts_cached_async,
)
from backend.modules.text_research.infrastructure.preprocessing import PreprocessingConfig
from backend.modules.text_research.infrastructure.topic_models import (
    k_sweep,
    seed_stability,
    train_topic_model,
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _plan_holdout_split(
    units: list[Any],
    prepared: PreparedCorpusArtifact,
    *,
    holdout_fraction: float | None,
    holdout_unit_ids: list[str] | None,
    random_seed: int,
) -> tuple[list[Any], PreparedCorpusArtifact, list[str] | None, dict[str, Any]]:
    """Split train vs holdout units for optional LDA perplexity diagnostics."""
    if not holdout_fraction and not holdout_unit_ids:
        return units, prepared, None, {}

    if holdout_unit_ids:
        holdout_set = set(holdout_unit_ids)
        train_idx = [index for index, unit in enumerate(units) if unit.id not in holdout_set]
        holdout_idx = [index for index, unit in enumerate(units) if unit.id in holdout_set]
    else:
        assert holdout_fraction is not None
        n_holdout = max(1, min(len(units) - 1, int(len(units) * holdout_fraction)))
        rng = random.Random(random_seed)
        holdout_idx = sorted(rng.sample(range(len(units)), n_holdout))
        holdout_set = set(holdout_idx)
        train_idx = [index for index in range(len(units)) if index not in holdout_set]

    if not train_idx or not holdout_idx:
        raise ValueError("Holdout split must leave at least one train and one holdout unit")

    def _slice_prepared(indices: list[int]) -> PreparedCorpusArtifact:
        selected_ids = {prepared.unit_ids[index] for index in indices}
        return build_prepared_artifact(
            [prepared.unit_ids[index] for index in indices],
            [prepared.original_units[index] for index in indices],
            [list(prepared.token_sequences[index]) for index in indices],
            prepared.preprocessing_profile,
            metadata_by_unit={
                key: value
                for key, value in prepared.metadata_by_unit.items()
                if key in selected_ids
            },
            document_ids=[prepared.document_ids[index] for index in indices]
            if prepared.document_ids
            else None,
            cleaned_texts=[prepared.cleaned_units[index] for index in indices],
            provenance=prepared.provenance,
        )

    train_units = [units[index] for index in train_idx]
    train_prepared = _slice_prepared(train_idx)
    holdout_texts = [prepared.texts_joined[index] for index in holdout_idx]
    meta = {
        "holdout_unit_ids": [units[index].id for index in holdout_idx],
        "n_holdout": len(holdout_idx),
        "n_train": len(train_idx),
    }
    return train_units, train_prepared, holdout_texts, meta


def build_metadata_breakdowns(
    dominant: list[int],
    units: list[Any],
    documents: dict[str, Any],
    group_by: list[str] | None,
) -> dict[str, dict[str, dict[str, int]]]:
    """Break down dominant topics by user-selected document metadata fields."""
    if not group_by:
        return {}
    breakdowns: dict[str, dict[str, dict[str, int]]] = {}
    for field in group_by:
        breakdown: dict[str, Counter[str]] = {}
        for index, topic_id in enumerate(dominant):
            document = documents.get(units[index].corpus_document_id)
            if document is None:
                continue
            value = document.get_field_value(field)
            if value is not None and value != "":
                breakdown.setdefault(str(value), Counter())[str(topic_id)] += 1
        if breakdown:
            breakdowns[field] = {value: dict(counts) for value, counts in breakdown.items()}
    return breakdowns


class TopicModelService(ResearchAccessMixin):
    async def _select_texts(
        self, corpus_id: str, *, unit_type: str, filters: dict[str, Any] | None
    ) -> list:
        filter_kwargs = _filter_kwargs(filters or {})
        documents = await self.repo.list_documents(corpus_id, **filter_kwargs)
        doc_ids = [document.id for document in documents]
        return await self.repo.list_text_units_for_corpus(
            corpus_id, unit_type=unit_type, document_ids=doc_ids if filter_kwargs else None
        )

    async def train(
        self,
        corpus_id: str,
        *,
        user_id: str,
        unit_type: str,
        algorithm: str = "lda",
        n_topics: int = 5,
        preprocessing_profile_id: str | None = None,
        max_iterations: int = 25,
        random_seed: int = 42,
        group_by: list[str] | None = None,
        holdout_fraction: float | None = None,
        holdout_unit_ids: list[str] | None = None,
        embedding_provider: str | None = None,
        embedding_model_name: str | None = None,
        persist_embedding_artifacts: bool = True,
        run_async: bool = True,
        **filters: Any,
    ) -> AnalysisRun:
        corpus = await self.get_corpus_or_404(corpus_id, user_id=user_id)
        units = await self._select_texts(corpus_id, unit_type=unit_type, filters=filters)
        if not units:
            raise HTTPException(
                status_code=422,
                detail="No text units match the requested corpus/unit_type/filters",
            )

        params = {
            "unit_type": unit_type,
            "algorithm": algorithm,
            "n_topics": n_topics,
            "preprocessing_profile_id": preprocessing_profile_id,
            "max_iterations": max_iterations,
            "random_seed": random_seed,
            "group_by": group_by,
            "holdout_fraction": holdout_fraction,
            "holdout_unit_ids": holdout_unit_ids,
            "embedding_provider": embedding_provider or "hashing",
            "embedding_model_name": embedding_model_name,
            "persist_embedding_artifacts": persist_embedding_artifacts,
            "filters": filters,
        }
        spec = build_spec_from_request(
            "topic_model",
            corpus_id,
            unit_type=unit_type,
            filters=filters,
            preprocessing_profile_id=preprocessing_profile_id,
            analysis_parameters={
                "algorithm": algorithm,
                "n_topics": n_topics,
                "max_iterations": max_iterations,
                "group_by": group_by,
                "embedding_provider": params["embedding_provider"],
            },
            random_seed=random_seed,
        )
        params = attach_run_identity(params, spec)
        run = await self.repo.create_run(
            AnalysisRun(
                project_id=corpus.project_id,
                corpus_id=corpus_id,
                run_type=AnalysisRunType.TOPIC_MODEL.value,
                status=AnalysisRunStatus.QUEUED.value,
                parameters_json=dumps(params),
                random_seed=random_seed,
                created_by=user_id,
            )
        )
        await self.db.commit()

        if run_async:
            from backend.modules.text_research.application.execution_service import ExecutionService

            await ExecutionService.submit(
                db=self.db, run=run, operation="topic_training", user_id=user_id
            )
        else:
            await self.execute_training(run.id)

        refreshed = await self.repo.get_run(run.id)
        assert refreshed is not None
        return refreshed

    async def execute_training(self, run_id: str) -> AnalysisRun:
        run = await self.repo.get_run(run_id)
        if run is None:
            raise ValueError(f"AnalysisRun {run_id} not found")
        if run.status in {AnalysisRunStatus.COMPLETED.value, AnalysisRunStatus.CANCELLED.value}:
            return run
        params = loads(run.parameters_json, {})

        await self.repo.update_run(
            run,
            status=AnalysisRunStatus.RUNNING.value,
            progress_stage="vectorizing",
            started_at=_utcnow(),
        )
        await self.db.commit()

        try:
            units = await self._select_texts(
                run.corpus_id, unit_type=params["unit_type"], filters=params.get("filters")
            )
            texts = [u.text for u in units]

            config = PreprocessingConfig().to_dict()
            if params.get("preprocessing_profile_id"):
                profile = await self.repo.get_preprocessing_profile(
                    params["preprocessing_profile_id"]
                )
                if profile is not None:
                    config.update(loads(profile.config_json, {}))

            prepared = await prepare_texts_cached_async(
                texts,
                config,
                corpus_id=run.corpus_id,
                unit_type=params["unit_type"],
                unit_ids=[unit.id for unit in units],
                document_ids=[unit.corpus_document_id for unit in units],
                filters=params.get("filters"),
                operation_config={"mode": "topic_training"},
            )
            holdout_meta: dict[str, Any] = {}
            holdout_texts: list[str] | None = None
            train_units = units
            train_prepared = prepared
            if params.get("holdout_unit_ids") or params.get("holdout_fraction"):
                train_units, train_prepared, holdout_texts, holdout_meta = _plan_holdout_split(
                    units,
                    prepared,
                    holdout_fraction=params.get("holdout_fraction"),
                    holdout_unit_ids=params.get("holdout_unit_ids"),
                    random_seed=params["random_seed"],
                )

            await self.repo.update_run(run, progress_stage="training")
            await self.db.commit()
            result = train_topic_model(
                [unit.text for unit in train_units],
                algorithm=params["algorithm"],
                n_topics=params["n_topics"],
                config=config,
                random_seed=params["random_seed"],
                max_iter=params["max_iterations"],
                prepared=train_prepared,
                holdout_texts=holdout_texts,
                embedding_provider=params.get("embedding_provider") or "hashing",
                embedding_model_name=params.get("embedding_model_name"),
                persist_embedding_artifacts=bool(params.get("persist_embedding_artifacts", True)),
            )

            await self.repo.update_run(run, progress_stage="saving")
            await self.db.commit()
            model_path, model_artifact_metadata = model_storage.save_artifact_with_metadata(
                result["model"], category="topic_models"
            )
            vectorizer_path, vectorizer_artifact_metadata = (
                model_storage.save_artifact_with_metadata(
                    result["vectorizer"], category="topic_vectorizers"
                )
            )

            doc_topic = result["doc_topic_distribution"]
            dominant = result["dominant_topics"]
            documents = {
                document.id: document for document in await self.repo.list_documents(run.corpus_id)
            }
            doc_topic_rows = [
                {
                    "text_unit_id": unit.id,
                    "topic_distribution": doc_topic[i],
                    "dominant_topic": dominant[i],
                }
                for i, unit in enumerate(train_units)
            ]
            distribution_path, distribution_artifact_metadata = (
                model_storage.save_artifact_with_metadata(
                    doc_topic_rows, category="topic_distributions"
                )
            )
            dominant_counts = {str(k): v for k, v in Counter(dominant).items()}
            topic_prevalence = {
                str(topic_id): sum(float(distribution[topic_id]) for distribution in doc_topic)
                / len(doc_topic)
                for topic_id in range(result["n_topics"])
            }
            representative_units: dict[str, list[dict[str, Any]]] = {}
            for topic_id in range(result["n_topics"]):
                ranked = sorted(
                    enumerate(doc_topic), key=lambda item: float(item[1][topic_id]), reverse=True
                )[:3]
                representative_units[str(topic_id)] = [
                    {
                        "text_unit_id": train_units[index].id,
                        "corpus_document_id": train_units[index].corpus_document_id,
                        "document_title": documents.get(train_units[index].corpus_document_id).title
                        if train_units[index].corpus_document_id in documents
                        else None,
                        "text": train_units[index].text,
                        "weight": float(distribution[topic_id]),
                    }
                    for index, distribution in ranked
                ]
            metadata_breakdowns = build_metadata_breakdowns(
                dominant,
                train_units,
                documents,
                params.get("group_by"),
            )

            await self.repo.update_run(
                run,
                status=AnalysisRunStatus.COMPLETED.value,
                progress_stage="completed",
                completed_at=_utcnow(),
                artifact_path=model_path,
                metrics_json=dumps({"n_topics": result["n_topics"], **result["diagnostics"]}),
                results_json=dumps(
                    {
                        "topics": result["topics"],
                        "family": result.get("family", "classical"),
                        "algorithm": result.get("algorithm"),
                        "components": result.get("components"),
                        "notes": result.get("notes") or result.get("diagnostics", {}).get("notes"),
                        "dominant_topic_counts": dominant_counts,
                        "topic_prevalence": topic_prevalence,
                        "representative_units": representative_units,
                        "metadata_breakdowns": metadata_breakdowns,
                        "holdout": holdout_meta or None,
                        "diagnostics": result["diagnostics"],
                        "corpus_checksum": prepared.corpus_checksum,
                        "pipeline_checksum": prepared.pipeline_checksum,
                        "analysis_spec_hash": params.get("analysis_spec_hash"),
                        "model_artifact_path": model_path,
                        "vectorizer_artifact_path": vectorizer_path,
                        "artifact_metadata": {
                            "model": model_artifact_metadata,
                            "vectorizer": vectorizer_artifact_metadata,
                            "topic_distribution": distribution_artifact_metadata,
                        },
                        "topic_distribution_artifact_path": distribution_path,
                        "unit_count": len(units),
                    }
                ),
            )
            await self.db.commit()
        except Exception as exc:  # noqa: BLE001
            await self.repo.update_run(
                run,
                status=AnalysisRunStatus.FAILED.value,
                completed_at=_utcnow(),
                error_message=str(exc),
            )
            await self.db.commit()
            raise

        refreshed = await self.repo.get_run(run_id)
        assert refreshed is not None
        return refreshed

    async def _resolve_training_texts_and_config(
        self,
        corpus_id: str,
        *,
        user_id: str,
        unit_type: str,
        preprocessing_profile_id: str | None,
        filters: dict[str, Any] | None,
    ) -> tuple[Any, Any, dict[str, Any]]:
        corpus = await self.get_corpus_or_404(corpus_id, user_id=user_id)
        units = await self._select_texts(corpus_id, unit_type=unit_type, filters=filters)
        if not units:
            raise HTTPException(
                status_code=422,
                detail="No text units match the requested corpus/unit_type/filters",
            )
        config = PreprocessingConfig().to_dict()
        if preprocessing_profile_id:
            profile = await self.repo.get_preprocessing_profile(preprocessing_profile_id)
            if profile is not None:
                config.update(loads(profile.config_json, {}))
        texts = [u.text for u in units]
        prepared = prepare_texts(
            texts,
            config,
            unit_ids=[u.id for u in units],
            document_ids=[u.corpus_document_id for u in units],
        )
        return corpus, prepared, config

    async def run_k_sweep(
        self,
        corpus_id: str,
        *,
        user_id: str,
        unit_type: str,
        k_values: list[int],
        algorithm: str = "lda",
        preprocessing_profile_id: str | None = None,
        max_iterations: int = 25,
        random_seed: int = 42,
        holdout_fraction: float | None = None,
        holdout_unit_ids: list[str] | None = None,
        run_async: bool = True,
        **filters: Any,
    ) -> AnalysisRun:
        """Create a queued K-sweep run; execute it outside the request worker."""
        corpus = await self.get_corpus_or_404(corpus_id, user_id=user_id)
        run = await self.repo.create_run(
            AnalysisRun(
                project_id=corpus.project_id,
                corpus_id=corpus_id,
                run_type=AnalysisRunType.TOPIC_MODEL.value,
                status=AnalysisRunStatus.QUEUED.value,
                progress_stage="queued",
                parameters_json=dumps(
                    {
                        "mode": "k_sweep",
                        "unit_type": unit_type,
                        "algorithm": algorithm,
                        "k_values": k_values,
                        "preprocessing_profile_id": preprocessing_profile_id,
                        "max_iterations": max_iterations,
                        "random_seed": random_seed,
                        "holdout_fraction": holdout_fraction,
                        "holdout_unit_ids": holdout_unit_ids,
                        "filters": filters,
                    }
                ),
                random_seed=random_seed,
                created_by=user_id,
            )
        )
        await self.db.commit()
        if run_async:
            from backend.modules.text_research.application.execution_service import ExecutionService

            await ExecutionService.submit(
                db=self.db, run=run, operation="topic_k_sweep", user_id=user_id
            )
        else:
            await self.execute_k_sweep(run.id)
        refreshed = await self.repo.get_run(run.id)
        assert refreshed is not None
        return refreshed

    async def execute_k_sweep(self, run_id: str) -> AnalysisRun:
        """Perform a persisted K sweep in a worker-owned database session."""
        run = await self.repo.get_run(run_id)
        if run is None:
            raise ValueError(f"AnalysisRun {run_id} not found")
        if run.status in {AnalysisRunStatus.COMPLETED.value, AnalysisRunStatus.CANCELLED.value}:
            return run
        params = loads(run.parameters_json, {})
        await self.repo.update_run(
            run,
            status=AnalysisRunStatus.RUNNING.value,
            progress_stage="preparing",
            started_at=_utcnow(),
        )
        await self.db.commit()
        try:
            units = await self._select_texts(
                run.corpus_id,
                unit_type=params["unit_type"],
                filters=params.get("filters"),
            )
            if not units:
                raise ValueError("No text units match the requested corpus/unit_type/filters")
            config = PreprocessingConfig().to_dict()
            if params.get("preprocessing_profile_id"):
                profile = await self.repo.get_preprocessing_profile(
                    params["preprocessing_profile_id"]
                )
                if profile is not None:
                    config.update(loads(profile.config_json, {}))
            prepared = prepare_texts(
                [unit.text for unit in units],
                config,
                unit_ids=[unit.id for unit in units],
                document_ids=[unit.corpus_document_id for unit in units],
            )
            holdout_meta: dict[str, Any] = {}
            holdout_texts: list[str] | None = None
            train_prepared = prepared
            train_texts = list(prepared.original_units)
            if params.get("holdout_unit_ids") or params.get("holdout_fraction"):
                _, train_prepared, holdout_texts, holdout_meta = _plan_holdout_split(
                    units,
                    prepared,
                    holdout_fraction=params.get("holdout_fraction"),
                    holdout_unit_ids=params.get("holdout_unit_ids"),
                    random_seed=params["random_seed"],
                )
                train_texts = list(train_prepared.original_units)
            await self.repo.update_run(run, progress_stage="training")
            await self.db.commit()
            rows = k_sweep(
                train_texts,
                params["k_values"],
                algorithm=params["algorithm"],
                config=config,
                random_seed=params["random_seed"],
                max_iter=int(params.get("max_iterations") or 25),
                prepared=train_prepared,
                holdout_texts=holdout_texts,
            )
            await self.repo.update_run(
                run,
                status=AnalysisRunStatus.COMPLETED.value,
                progress_stage="completed",
                completed_at=_utcnow(),
                metrics_json=dumps(
                    {"unit_count": len(prepared.unit_ids), "k_values": params["k_values"]}
                ),
                results_json=dumps(
                    {
                        "rows": rows,
                        "holdout": holdout_meta or None,
                        "corpus_checksum": prepared.corpus_checksum,
                        "pipeline_checksum": prepared.pipeline_checksum,
                    }
                ),
            )
            await self.db.commit()
        except Exception as exc:  # noqa: BLE001
            await self.repo.update_run(
                run,
                status=AnalysisRunStatus.FAILED.value,
                completed_at=_utcnow(),
                error_message=str(exc),
            )
            await self.db.commit()
            raise
        refreshed = await self.repo.get_run(run_id)
        assert refreshed is not None
        return refreshed

    async def run_seed_stability(
        self,
        corpus_id: str,
        *,
        user_id: str,
        unit_type: str,
        seeds: list[int],
        algorithm: str = "lda",
        n_topics: int = 5,
        preprocessing_profile_id: str | None = None,
        max_iterations: int = 25,
        run_async: bool = True,
        **filters: Any,
    ) -> AnalysisRun:
        """Create a queued multi-seed topic-stability run."""
        corpus = await self.get_corpus_or_404(corpus_id, user_id=user_id)
        run = await self.repo.create_run(
            AnalysisRun(
                project_id=corpus.project_id,
                corpus_id=corpus_id,
                run_type=AnalysisRunType.TOPIC_MODEL.value,
                status=AnalysisRunStatus.QUEUED.value,
                progress_stage="queued",
                parameters_json=dumps(
                    {
                        "mode": "seed_stability",
                        "unit_type": unit_type,
                        "algorithm": algorithm,
                        "n_topics": n_topics,
                        "seeds": seeds,
                        "preprocessing_profile_id": preprocessing_profile_id,
                        "max_iterations": max_iterations,
                        "filters": filters,
                    }
                ),
                created_by=user_id,
            )
        )
        await self.db.commit()
        if run_async:
            from backend.modules.text_research.application.execution_service import ExecutionService

            await ExecutionService.submit(
                db=self.db,
                run=run,
                operation="topic_seed_stability",
                user_id=user_id,
            )
        else:
            await self.execute_seed_stability(run.id)
        refreshed = await self.repo.get_run(run.id)
        assert refreshed is not None
        return refreshed

    async def execute_seed_stability(self, run_id: str) -> AnalysisRun:
        """Perform a persisted multi-seed stability diagnostic in a worker."""
        run = await self.repo.get_run(run_id)
        if run is None:
            raise ValueError(f"AnalysisRun {run_id} not found")
        if run.status in {AnalysisRunStatus.COMPLETED.value, AnalysisRunStatus.CANCELLED.value}:
            return run
        params = loads(run.parameters_json, {})
        await self.repo.update_run(
            run,
            status=AnalysisRunStatus.RUNNING.value,
            progress_stage="preparing",
            started_at=_utcnow(),
        )
        await self.db.commit()
        try:
            units = await self._select_texts(
                run.corpus_id,
                unit_type=params["unit_type"],
                filters=params.get("filters"),
            )
            if not units:
                raise ValueError("No text units match the requested corpus/unit_type/filters")
            config = PreprocessingConfig().to_dict()
            if params.get("preprocessing_profile_id"):
                profile = await self.repo.get_preprocessing_profile(
                    params["preprocessing_profile_id"]
                )
                if profile is not None:
                    config.update(loads(profile.config_json, {}))
            prepared = prepare_texts(
                [unit.text for unit in units],
                config,
                unit_ids=[unit.id for unit in units],
                document_ids=[unit.corpus_document_id for unit in units],
            )
            await self.repo.update_run(run, progress_stage="training")
            await self.db.commit()
            stability = seed_stability(
                list(prepared.original_units),
                params["seeds"],
                algorithm=params["algorithm"],
                n_topics=params["n_topics"],
                config=config,
                max_iter=int(params.get("max_iterations") or 25),
                prepared=prepared,
            )
            await self.repo.update_run(
                run,
                status=AnalysisRunStatus.COMPLETED.value,
                progress_stage="completed",
                completed_at=_utcnow(),
                metrics_json=dumps(
                    {
                        "unit_count": len(prepared.unit_ids),
                        "mean_stability_jaccard": stability["mean_stability_jaccard"],
                    }
                ),
                results_json=dumps(
                    {
                        **stability,
                        "corpus_checksum": prepared.corpus_checksum,
                        "pipeline_checksum": prepared.pipeline_checksum,
                    }
                ),
            )
            await self.db.commit()
        except Exception as exc:  # noqa: BLE001
            await self.repo.update_run(
                run,
                status=AnalysisRunStatus.FAILED.value,
                completed_at=_utcnow(),
                error_message=str(exc),
            )
            await self.db.commit()
            raise
        refreshed = await self.repo.get_run(run_id)
        assert refreshed is not None
        return refreshed

    async def name_topic(
        self, run_id: str, *, user_id: str, topic_id: int, human_name: str
    ) -> TopicLabel:
        await self.get_run_or_404(run_id, user_id=user_id)
        label = await self.repo.upsert_topic_label(
            analysis_run_id=run_id, topic_id=topic_id, human_name=human_name, created_by=user_id
        )
        await self.db.commit()
        return label

    async def get_topic_labels(self, run_id: str, *, user_id: str) -> list[TopicLabel]:
        await self.get_run_or_404(run_id, user_id=user_id)
        return await self.repo.list_topic_labels(run_id)
