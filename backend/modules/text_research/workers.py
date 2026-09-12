"""Celery dispatch helpers for long-running text research jobs."""

from __future__ import annotations

import logging

from backend.core.config import settings
from backend.modules.text_research.domain.analysis_task import resource_class_for
from backend.workers.async_dispatch import (
    dispatch_background_sync_job,
    run_async_in_sync_context,
)

logger = logging.getLogger(__name__)

_RESOURCE_CLASS_QUEUES = {
    "research_light": lambda: settings.RESEARCH_QUEUE_LIGHT,
    "research_cpu": lambda: settings.RESEARCH_QUEUE_CPU,
    "research_io": lambda: settings.RESEARCH_QUEUE_IO,
    "research_nlp": lambda: settings.RESEARCH_QUEUE_NLP,
    "research_memory": lambda: settings.RESEARCH_QUEUE_MEMORY,
    "research_gpu": lambda: settings.RESEARCH_QUEUE_GPU,
}


def queue_for_resource_class(resource_class: str) -> str:
    """Resolve Celery queue name for a worker resource class."""
    resolver = _RESOURCE_CLASS_QUEUES.get(resource_class)
    if resolver is None:
        return settings.CELERY_TASK_DEFAULT_QUEUE
    return resolver()


def _ensure_worker_parallelism() -> None:
    """Apply thread caps for Celery workers and eager in-process research jobs."""
    from backend.workers.parallelism import configure_worker_parallelism

    configure_worker_parallelism(force=False)


def _run_with_session(coro_factory):
    from backend.db.session import SessionLocal

    async def _run() -> None:
        async with SessionLocal() as db:
            await coro_factory(db)

    _ensure_worker_parallelism()
    run_async_in_sync_context(_run())


def segmentation_sync(*, run_id: str, user_id: str) -> None:
    from backend.modules.text_research.application.segmentation_service import SegmentationService

    async def _execute(db):
        service = SegmentationService(db)
        await service.execute_segmentation(run_id)

    try:
        logger.info("Research segmentation started run=%s user=%s", run_id, user_id)
        _run_with_session(_execute)
        logger.info("Research segmentation completed run=%s", run_id)
    except Exception:
        logger.exception("Research segmentation failed run=%s user=%s", run_id, user_id)
        raise


def classifier_training_sync(*, run_id: str, user_id: str) -> None:
    from backend.modules.text_research.application.classification_service import (
        ClassificationService,
    )

    async def _execute(db):
        service = ClassificationService(db)
        await service.execute_training(run_id)

    try:
        logger.info("Classifier training started run=%s user=%s", run_id, user_id)
        _run_with_session(_execute)
        logger.info("Classifier training completed run=%s", run_id)
    except Exception:
        logger.exception("Classifier training failed run=%s user=%s", run_id, user_id)
        raise


def topic_model_training_sync(*, run_id: str, user_id: str) -> None:
    from backend.modules.text_research.application.topic_model_service import TopicModelService

    async def _execute(db):
        service = TopicModelService(db)
        await service.execute_training(run_id)

    try:
        logger.info("Topic model training started run=%s user=%s", run_id, user_id)
        _run_with_session(_execute)
        logger.info("Topic model training completed run=%s", run_id)
    except Exception:
        logger.exception("Topic model training failed run=%s user=%s", run_id, user_id)
        raise


def topic_k_sweep_sync(*, run_id: str, user_id: str) -> None:
    from backend.modules.text_research.application.topic_model_service import TopicModelService

    async def _execute(db):
        await TopicModelService(db).execute_k_sweep(run_id)

    logger.info("Topic K sweep started run=%s user=%s", run_id, user_id)
    _run_with_session(_execute)
    logger.info("Topic K sweep completed run=%s", run_id)


def topic_seed_stability_sync(*, run_id: str, user_id: str) -> None:
    from backend.modules.text_research.application.topic_model_service import TopicModelService

    async def _execute(db):
        await TopicModelService(db).execute_seed_stability(run_id)

    logger.info("Topic seed stability started run=%s user=%s", run_id, user_id)
    _run_with_session(_execute)
    logger.info("Topic seed stability completed run=%s", run_id)


def robustness_sweep_sync(*, run_id: str, user_id: str) -> None:
    from backend.modules.text_research.application.robustness_service import RobustnessService

    async def _execute(db):
        service = RobustnessService(db)
        await service.execute_sweep(run_id)

    try:
        logger.info("Robustness sweep started run=%s user=%s", run_id, user_id)
        _run_with_session(_execute)
        logger.info("Robustness sweep completed run=%s", run_id)
    except Exception:
        logger.exception("Robustness sweep failed run=%s user=%s", run_id, user_id)
        raise


def prediction_sync(*, run_id: str, user_id: str) -> None:
    from backend.modules.text_research.application.prediction_service import PredictionService

    async def _execute(db):
        await PredictionService(db).execute_prediction(run_id)

    _run_with_session(_execute)


def quantitative_analysis_sync(*, run_id: str, user_id: str) -> None:
    from backend.modules.text_research.application.quantitative_analysis_service import (
        QuantitativeAnalysisService,
    )

    async def _execute(db):
        await QuantitativeAnalysisService(db).execute_quantitative(run_id)

    try:
        logger.info("Quantitative analysis started run=%s user=%s", run_id, user_id)
        _run_with_session(_execute)
        logger.info("Quantitative analysis completed run=%s", run_id)
    except Exception:
        logger.exception("Quantitative analysis failed run=%s user=%s", run_id, user_id)
        raise


def corpus_synthesis_sync(*, run_id: str, user_id: str) -> None:
    from backend.modules.text_research.application.corpus_synthesis_service import (
        CorpusSynthesisService,
    )

    async def _execute(db):
        await CorpusSynthesisService(db).execute_synthesis(run_id)

    try:
        logger.info("Corpus synthesis started run=%s user=%s", run_id, user_id)
        _run_with_session(_execute)
        logger.info("Corpus synthesis completed run=%s", run_id)
    except Exception:
        logger.exception("Corpus synthesis failed run=%s user=%s", run_id, user_id)
        raise


def queue_segmentation(*, run_id: str, user_id: str, task_id: str | None = None) -> str | None:
    from backend.workers.tasks import research_segmentation_task

    return dispatch_background_sync_job(
        target=segmentation_sync,
        kwargs={"run_id": run_id, "user_id": user_id},
        celery_task=research_segmentation_task,
        celery_kwargs={"run_id": run_id, "user_id": user_id},
        queue=queue_for_resource_class("research_io"),
        job_name="research-segmentation",
        task_id=task_id,
    )


def queue_classifier_training(*, run_id: str, user_id: str, task_id: str | None = None) -> str | None:
    from backend.workers.tasks import research_classifier_training_task

    return dispatch_background_sync_job(
        target=classifier_training_sync,
        kwargs={"run_id": run_id, "user_id": user_id},
        celery_task=research_classifier_training_task,
        celery_kwargs={"run_id": run_id, "user_id": user_id},
        queue=queue_for_resource_class(resource_class_for("classification")),
        job_name="research-classifier-training",
        task_id=task_id,
    )


def queue_topic_model_training(*, run_id: str, user_id: str, task_id: str | None = None) -> str | None:
    from backend.workers.tasks import research_topic_model_training_task

    return dispatch_background_sync_job(
        target=topic_model_training_sync,
        kwargs={"run_id": run_id, "user_id": user_id},
        celery_task=research_topic_model_training_task,
        celery_kwargs={"run_id": run_id, "user_id": user_id},
        queue=queue_for_resource_class(resource_class_for("topic_model")),
        job_name="research-topic-model",
        task_id=task_id,
    )


def queue_topic_k_sweep(*, run_id: str, user_id: str, task_id: str | None = None) -> str | None:
    from backend.workers.tasks import research_topic_k_sweep_task

    return dispatch_background_sync_job(
        target=topic_k_sweep_sync,
        kwargs={"run_id": run_id, "user_id": user_id},
        celery_task=research_topic_k_sweep_task,
        celery_kwargs={"run_id": run_id, "user_id": user_id},
        queue=queue_for_resource_class(resource_class_for("topic_model")),
        job_name="research-topic-k-sweep",
        task_id=task_id,
    )


def queue_topic_seed_stability(*, run_id: str, user_id: str, task_id: str | None = None) -> str | None:
    from backend.workers.tasks import research_topic_seed_stability_task

    return dispatch_background_sync_job(
        target=topic_seed_stability_sync,
        kwargs={"run_id": run_id, "user_id": user_id},
        celery_task=research_topic_seed_stability_task,
        celery_kwargs={"run_id": run_id, "user_id": user_id},
        queue=queue_for_resource_class(resource_class_for("topic_model")),
        job_name="research-topic-seed-stability",
        task_id=task_id,
    )


def queue_robustness_sweep(*, run_id: str, user_id: str, task_id: str | None = None) -> str | None:
    from backend.workers.tasks import research_robustness_sweep_task

    return dispatch_background_sync_job(
        target=robustness_sweep_sync,
        kwargs={"run_id": run_id, "user_id": user_id},
        celery_task=research_robustness_sweep_task,
        celery_kwargs={"run_id": run_id, "user_id": user_id},
        queue=queue_for_resource_class(resource_class_for("classification")),
        job_name="research-robustness-sweep",
        task_id=task_id,
    )


def queue_prediction(*, run_id: str, user_id: str, task_id: str | None = None) -> str | None:
    from backend.workers.tasks import research_prediction_task

    return dispatch_background_sync_job(
        target=prediction_sync,
        kwargs={"run_id": run_id, "user_id": user_id},
        celery_task=research_prediction_task,
        celery_kwargs={"run_id": run_id, "user_id": user_id},
        queue=queue_for_resource_class(resource_class_for("classification")),
        job_name="research-prediction",
        task_id=task_id,
    )


def queue_quantitative_analysis(*, run_id: str, user_id: str, task_id: str | None = None) -> str | None:
    from backend.workers.tasks import research_quantitative_analysis_task

    return dispatch_background_sync_job(
        target=quantitative_analysis_sync,
        kwargs={"run_id": run_id, "user_id": user_id},
        celery_task=research_quantitative_analysis_task,
        celery_kwargs={"run_id": run_id, "user_id": user_id},
        queue=queue_for_resource_class("research_cpu"),
        job_name="research-quantitative-analysis",
        task_id=task_id,
    )


def queue_corpus_synthesis(*, run_id: str, user_id: str, task_id: str | None = None) -> str | None:
    from backend.workers.tasks import research_corpus_synthesis_task

    return dispatch_background_sync_job(
        target=corpus_synthesis_sync,
        kwargs={"run_id": run_id, "user_id": user_id},
        celery_task=research_corpus_synthesis_task,
        celery_kwargs={"run_id": run_id, "user_id": user_id},
        queue=queue_for_resource_class("research_cpu"),
        job_name="research-corpus-synthesis",
        task_id=task_id,
    )


def queue_research_operation(
    *, operation: str, run_id: str, user_id: str, task_id: str | None = None
) -> str | None:
    """Dispatch a named persisted operation through the shared worker boundary."""
    dispatchers = {
        "segmentation": queue_segmentation,
        "classification": queue_classifier_training,
        "topic_training": queue_topic_model_training,
        "topic_k_sweep": queue_topic_k_sweep,
        "topic_seed_stability": queue_topic_seed_stability,
        "robustness": queue_robustness_sweep,
        "prediction": queue_prediction,
        "quantitative": queue_quantitative_analysis,
        "corpus_synthesis": queue_corpus_synthesis,
    }
    try:
        dispatcher = dispatchers[operation]
    except KeyError as exc:
        raise ValueError(f"Unsupported research operation {operation!r}") from exc
    return dispatcher(run_id=run_id, user_id=user_id, task_id=task_id)
