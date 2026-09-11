from celery import Celery

from backend.core.config import settings
from backend.core.logging import setup_logging

setup_logging()

# Cap BLAS/OpenMP early in the worker parent process (children re-apply on
# worker_process_init). Safe no-op when RESEARCH_APPLY_THREAD_LIMITS=false.
from backend.workers.parallelism import configure_worker_parallelism  # noqa: E402

configure_worker_parallelism(force=False, overwrite_env=False)

celery_app = Celery(
    "app_backend",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["backend.workers.tasks"],
)

celery_app.conf.update(
    task_default_queue=settings.CELERY_TASK_DEFAULT_QUEUE,
    task_routes={
        "backend.workers.tasks.send_email_task": {"queue": settings.CELERY_EMAIL_QUEUE},
        "backend.workers.tasks.index_rag_document_task": {
            "queue": settings.CELERY_TASK_DEFAULT_QUEUE
        },
        "backend.workers.tasks.run_ai_evaluation_task": {
            "queue": settings.CELERY_TASK_DEFAULT_QUEUE
        },
        "backend.workers.tasks.research_segmentation_task": {"queue": settings.RESEARCH_QUEUE_IO},
        "backend.workers.tasks.research_classifier_training_task": {
            "queue": settings.RESEARCH_QUEUE_CPU
        },
        "backend.workers.tasks.research_topic_model_training_task": {
            "queue": settings.RESEARCH_QUEUE_GPU
        },
        "backend.workers.tasks.research_topic_k_sweep_task": {"queue": settings.RESEARCH_QUEUE_GPU},
        "backend.workers.tasks.research_topic_seed_stability_task": {
            "queue": settings.RESEARCH_QUEUE_GPU
        },
        "backend.workers.tasks.research_robustness_sweep_task": {
            "queue": settings.RESEARCH_QUEUE_CPU
        },
        "backend.workers.tasks.research_prediction_task": {"queue": settings.RESEARCH_QUEUE_CPU},
        "backend.workers.tasks.research_quantitative_analysis_task": {
            "queue": settings.RESEARCH_QUEUE_CPU
        },
        "backend.workers.tasks.research_r_quantitative_analysis_task": {
            "queue": settings.RESEARCH_QUEUE_R
        },
        # Orchestrator only — Python child stays on research_cpu; R child on research_r.
        "backend.workers.tasks.research_engine_comparison_task": {
            "queue": settings.RESEARCH_QUEUE_CPU
        },
        "backend.workers.tasks.research_engine_comparison_finalize_task": {
            "queue": settings.RESEARCH_QUEUE_LIGHT
        },
    },
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    result_expires=settings.CELERY_RESULT_EXPIRES_SECONDS,
    broker_connection_retry_on_startup=True,
    task_track_started=True,
    task_ignore_result=True,
    timezone="UTC",
    enable_utc=True,
)

import backend.workers.logging_hooks  # noqa: F401,E402 — register Celery signal handlers
