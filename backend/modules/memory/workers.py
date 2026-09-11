from __future__ import annotations

import logging

from backend.core.cache import get_async_redis_client
from backend.core.config import settings
from backend.workers.async_dispatch import dispatch_background_sync_job, run_async_in_sync_context

logger = logging.getLogger(__name__)


def extract_turn_memories_sync(
    *,
    user_id: str,
    agent_id: str,
    run_id: str,
    project_id: str | None,
    user_message: str,
    assistant_message: str,
    source_message_id: str,
) -> None:
    from backend.db.session import SessionLocal
    from backend.modules.memory.application.memory_service import MemoryService

    async def _run() -> None:
        completion_key = f"memory:turn-extraction:complete:{source_message_id}"
        try:
            if await get_async_redis_client().get(completion_key):
                logger.info(
                    "Memory extraction already complete source_message=%s", source_message_id
                )
                return
        except Exception:
            logger.warning("Memory extraction idempotency cache unavailable", exc_info=True)
        async with SessionLocal() as db:
            service = MemoryService(db)
            await service.process_turn_memories(
                user_id=user_id,
                agent_id=agent_id,
                run_id=run_id,
                project_id=project_id,
                user_message=user_message,
                assistant_message=assistant_message,
                source_message_id=source_message_id,
            )
        try:
            await get_async_redis_client().set(completion_key, "1", ex=30 * 24 * 60 * 60)
        except Exception:
            logger.warning("Memory extraction completion cache unavailable", exc_info=True)

    run_async_in_sync_context(_run())


def queue_turn_memory_extraction(**payload: str | None) -> None:
    from backend.workers.tasks import extract_turn_memories_task

    kwargs = {
        "user_id": str(payload["user_id"]),
        "agent_id": str(payload["agent_id"]),
        "run_id": str(payload["run_id"]),
        "project_id": payload.get("project_id"),
        "user_message": str(payload["user_message"]),
        "assistant_message": str(payload["assistant_message"]),
        "source_message_id": str(payload["source_message_id"]),
    }
    dispatch_background_sync_job(
        target=extract_turn_memories_sync,
        kwargs=kwargs,
        celery_task=extract_turn_memories_task,
        celery_kwargs=kwargs,
        queue=settings.CELERY_TASK_DEFAULT_QUEUE,
        job_name="memory-extraction",
    )
    logger.info("Queued memory extraction source_message=%s", kwargs["source_message_id"])
