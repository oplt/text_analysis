"""Celery dispatch helpers for long-running text research jobs."""

from __future__ import annotations

import logging

from backend.core.config import settings
from backend.workers.async_dispatch import (
    dispatch_background_sync_job,
    run_async_in_sync_context,
)

logger = logging.getLogger(__name__)


def _run_with_session(coro_factory):
    from backend.db.session import SessionLocal

    async def _run() -> None:
        async with SessionLocal() as db:
            await coro_factory(db)

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
    from backend.modules.text_research.application.classification_service import ClassificationService

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


def queue_segmentation(*, run_id: str, user_id: str) -> None:
    from backend.workers.tasks import research_segmentation_task

    dispatch_background_sync_job(
        target=segmentation_sync,
        kwargs={"run_id": run_id, "user_id": user_id},
        celery_task=research_segmentation_task,
        celery_kwargs={"run_id": run_id, "user_id": user_id},
        queue=settings.CELERY_TASK_DEFAULT_QUEUE,
        job_name="research-segmentation",
    )


def queue_classifier_training(*, run_id: str, user_id: str) -> None:
    from backend.workers.tasks import research_classifier_training_task

    dispatch_background_sync_job(
        target=classifier_training_sync,
        kwargs={"run_id": run_id, "user_id": user_id},
        celery_task=research_classifier_training_task,
        celery_kwargs={"run_id": run_id, "user_id": user_id},
        queue=settings.CELERY_TASK_DEFAULT_QUEUE,
        job_name="research-classifier-training",
    )


def queue_topic_model_training(*, run_id: str, user_id: str) -> None:
    from backend.workers.tasks import research_topic_model_training_task

    dispatch_background_sync_job(
        target=topic_model_training_sync,
        kwargs={"run_id": run_id, "user_id": user_id},
        celery_task=research_topic_model_training_task,
        celery_kwargs={"run_id": run_id, "user_id": user_id},
        queue=settings.CELERY_TASK_DEFAULT_QUEUE,
        job_name="research-topic-model",
    )


def queue_robustness_sweep(*, run_id: str, user_id: str) -> None:
    from backend.workers.tasks import research_robustness_sweep_task

    dispatch_background_sync_job(
        target=robustness_sweep_sync,
        kwargs={"run_id": run_id, "user_id": user_id},
        celery_task=research_robustness_sweep_task,
        celery_kwargs={"run_id": run_id, "user_id": user_id},
        queue=settings.CELERY_TASK_DEFAULT_QUEUE,
        job_name="research-robustness-sweep",
    )


def queue_prediction(*, run_id: str, user_id: str) -> None:
    from backend.workers.tasks import research_prediction_task

    dispatch_background_sync_job(target=prediction_sync, kwargs={"run_id": run_id, "user_id": user_id}, celery_task=research_prediction_task, celery_kwargs={"run_id": run_id, "user_id": user_id}, queue="research_cpu", job_name="research-prediction")
