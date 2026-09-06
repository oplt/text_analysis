"""Topic modeling (LDA/NMF) training, persistence, and human topic naming."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from typing import Any

from fastapi import HTTPException

from backend.modules.text_research.application.access import ResearchAccessMixin
from backend.modules.text_research.application.quantitative_analysis_service import (
    _filter_kwargs,
)
from backend.modules.text_research.domain.enums import AnalysisRunStatus, AnalysisRunType
from backend.modules.text_research.domain.models import AnalysisRun, TopicLabel, dumps, loads
from backend.modules.text_research.infrastructure import model_storage
from backend.modules.text_research.infrastructure.preprocessing import PreprocessingConfig
from backend.modules.text_research.infrastructure.topic_models import train_topic_model


def _utcnow() -> datetime:
    return datetime.now(UTC)


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
        run_async: bool = False,
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
            "filters": filters,
        }
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
            from backend.modules.text_research.workers import queue_topic_model_training

            queue_topic_model_training(run_id=run.id, user_id=user_id)
        else:
            await self.execute_training(run.id)

        refreshed = await self.repo.get_run(run.id)
        assert refreshed is not None
        return refreshed

    async def execute_training(self, run_id: str) -> AnalysisRun:
        run = await self.repo.get_run(run_id)
        if run is None:
            raise ValueError(f"AnalysisRun {run_id} not found")
        params = loads(run.parameters_json, {})

        await self.repo.update_run(
            run, status=AnalysisRunStatus.RUNNING.value, progress_stage="vectorizing", started_at=_utcnow()
        )
        await self.db.commit()

        try:
            units = await self._select_texts(
                run.corpus_id, unit_type=params["unit_type"], filters=params.get("filters")
            )
            texts = [u.text for u in units]

            config = PreprocessingConfig().to_dict()
            if params.get("preprocessing_profile_id"):
                profile = await self.repo.get_preprocessing_profile(params["preprocessing_profile_id"])
                if profile is not None:
                    config.update(loads(profile.config_json, {}))

            await self.repo.update_run(run, progress_stage="training")
            await self.db.commit()
            result = train_topic_model(
                texts,
                algorithm=params["algorithm"],
                n_topics=params["n_topics"],
                config=config,
                random_seed=params["random_seed"],
                max_iter=params["max_iterations"],
            )

            await self.repo.update_run(run, progress_stage="saving")
            await self.db.commit()
            model_path, model_artifact_metadata = model_storage.save_artifact_with_metadata(
                result["model"], category="topic_models"
            )
            vectorizer_path, vectorizer_artifact_metadata = model_storage.save_artifact_with_metadata(
                result["vectorizer"], category="topic_vectorizers"
            )

            doc_topic = result["doc_topic_distribution"]
            dominant = result["dominant_topics"]
            documents = {document.id: document for document in await self.repo.list_documents(run.corpus_id)}
            doc_topic_rows = [
                {
                    "text_unit_id": unit.id,
                    "topic_distribution": doc_topic[i],
                    "dominant_topic": dominant[i],
                }
                for i, unit in enumerate(units)
            ]
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
                        "text_unit_id": units[index].id,
                        "corpus_document_id": units[index].corpus_document_id,
                        "document_title": documents.get(units[index].corpus_document_id).title
                        if units[index].corpus_document_id in documents
                        else None,
                        "text": units[index].text,
                        "weight": float(distribution[topic_id]),
                    }
                    for index, distribution in ranked
                ]
            metadata_breakdowns: dict[str, dict[str, dict[str, int]]] = {}
            for field in ("organization", "region", "language"):
                breakdown: dict[str, Counter[str]] = {}
                for index, topic_id in enumerate(dominant):
                    document = documents.get(units[index].corpus_document_id)
                    value = getattr(document, field, None) if document else None
                    if value:
                        breakdown.setdefault(str(value), Counter())[str(topic_id)] += 1
                if breakdown:
                    metadata_breakdowns[field] = {
                        value: dict(counts) for value, counts in breakdown.items()
                    }

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
                        "dominant_topic_counts": dominant_counts,
                        "topic_prevalence": topic_prevalence,
                        "representative_units": representative_units,
                        "metadata_breakdowns": metadata_breakdowns,
                        "model_artifact_path": model_path,
                        "vectorizer_artifact_path": vectorizer_path,
                        "artifact_metadata": {
                            "model": model_artifact_metadata,
                            "vectorizer": vectorizer_artifact_metadata,
                        },
                        "doc_topic": doc_topic_rows,
                        "unit_count": len(units),
                    }
                ),
            )
            await self.db.commit()
        except Exception as exc:  # noqa: BLE001
            await self.repo.update_run(
                run, status=AnalysisRunStatus.FAILED.value, completed_at=_utcnow(), error_message=str(exc)
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
