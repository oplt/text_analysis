from __future__ import annotations

import logging

from backend.core.config import settings
from backend.workers.async_dispatch import (
    dispatch_background_sync_job,
    run_async_in_sync_context,
)

logger = logging.getLogger(__name__)


def index_document_sync(
    *,
    document_id: str,
    user_id: str,
    job_id: str | None = None,
) -> None:
    from backend.db.session import SessionLocal
    from backend.modules.rag.application.document_ingestion_service import DocumentIngestionService

    async def _run() -> None:
        async with SessionLocal() as db:
            service = DocumentIngestionService(db)
            await service.index_document(
                document_id=document_id,
                user_id=user_id,
                file_content=None,
                job_id=job_id,
            )

    try:
        logger.info(
            "RAG indexing started document=%s user=%s job=%s",
            document_id,
            user_id,
            job_id,
        )
        run_async_in_sync_context(_run())
        logger.info(
            "RAG indexing completed document=%s user=%s job=%s",
            document_id,
            user_id,
            job_id,
        )
    except Exception:
        logger.exception(
            "RAG indexing failed document=%s user=%s job=%s",
            document_id,
            user_id,
            job_id,
        )
        raise


def queue_document_indexing(
    *,
    document_id: str,
    user_id: str,
    job_id: str | None = None,
) -> None:
    payload = {"document_id": document_id, "user_id": user_id}
    if job_id:
        payload["job_id"] = job_id

    from backend.workers.tasks import index_rag_document_task

    dispatch_background_sync_job(
        target=index_document_sync,
        kwargs={
            "document_id": document_id,
            "user_id": user_id,
            "job_id": job_id,
        },
        celery_task=index_rag_document_task,
        celery_kwargs=payload,
        queue=settings.CELERY_TASK_DEFAULT_QUEUE,
        job_name="rag-indexing",
    )
    logger.info("Queued RAG indexing for document=%s user=%s", document_id, user_id)


def cleanup_document_sync(
    *,
    document_id: str,
    user_id: str,
    storage_path: str | None,
) -> None:
    from backend.db.session import SessionLocal
    from backend.modules.rag.application.document_ingestion_service import DocumentIngestionService

    async def _run() -> None:
        async with SessionLocal() as db:
            await DocumentIngestionService(db).cleanup_deleted_document(
                document_id=document_id,
                user_id=user_id,
                storage_path=storage_path,
            )

    run_async_in_sync_context(_run())


def queue_document_cleanup(
    *,
    document_id: str,
    user_id: str,
    storage_path: str | None,
) -> None:
    from backend.workers.tasks import cleanup_rag_document_task

    payload = {
        "document_id": document_id,
        "user_id": user_id,
        "storage_path": storage_path,
    }
    dispatch_background_sync_job(
        target=cleanup_document_sync,
        kwargs=payload,
        celery_task=cleanup_rag_document_task,
        celery_kwargs=payload,
        queue=settings.CELERY_TASK_DEFAULT_QUEUE,
        job_name="rag-document-cleanup",
    )
    logger.info("Queued RAG cleanup document=%s user=%s", document_id, user_id)
