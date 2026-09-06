from __future__ import annotations

import logging

from backend.core.config import settings
from backend.workers.async_dispatch import dispatch_background_sync_job, run_async_in_sync_context

logger = logging.getLogger(__name__)


def run_evaluation_sync(
    *,
    evaluation_run_id: str,
    user_id: str,
    dataset_id: str,
    prompt_version_id: str,
) -> None:
    from backend.db.session import SessionLocal
    from backend.modules.ai.service import AiService

    async def _run() -> None:
        async with SessionLocal() as db:
            service = AiService(db)
            await service.execute_evaluation_run(
                evaluation_run_id=evaluation_run_id,
                user_id=user_id,
                dataset_id=dataset_id,
                prompt_version_id=prompt_version_id,
            )

    run_async_in_sync_context(_run())


def queue_evaluation_run(
    *,
    evaluation_run_id: str,
    user_id: str,
    dataset_id: str,
    prompt_version_id: str,
) -> None:
    payload = {
        "evaluation_run_id": evaluation_run_id,
        "user_id": user_id,
        "dataset_id": dataset_id,
        "prompt_version_id": prompt_version_id,
    }
    from backend.workers.tasks import run_ai_evaluation_task

    dispatch_background_sync_job(
        target=run_evaluation_sync,
        kwargs=payload,
        celery_task=run_ai_evaluation_task,
        celery_kwargs=payload,
        queue=settings.CELERY_TASK_DEFAULT_QUEUE,
        job_name="ai-evaluation",
    )
    logger.info(
        "Queued AI evaluation run %s for dataset=%s user=%s",
        evaluation_run_id,
        dataset_id,
        user_id,
    )
