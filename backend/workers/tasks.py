from backend.modules.memory.workers import extract_turn_memories_sync
from backend.modules.rag.workers import cleanup_document_sync, index_document_sync
from backend.modules.text_research.infrastructure.execution_policy import retry_policy_for
from backend.modules.text_research.workers import (
    classifier_training_sync,
    robustness_sweep_sync,
    prediction_sync,
    segmentation_sync,
    topic_k_sweep_sync,
    topic_model_training_sync,
    topic_seed_stability_sync,
)
from backend.workers.celery_app import celery_app
from backend.workers.email import send_email_sync
from backend.workers.evaluation import run_evaluation_sync


def _research_task_options(resource_class: str) -> dict:
    policy = retry_policy_for(resource_class)
    return {
        "bind": True,
        "autoretry_for": (Exception,),
        "retry_backoff": False,
        "retry_jitter": False,
        "max_retries": policy["max_retries"],
        "default_retry_delay": policy["countdown"],
    }


@celery_app.task(
    name="backend.workers.tasks.send_email_task",
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_jitter=True,
    max_retries=5,
)
def send_email_task(
    *,
    to: str,
    subject: str,
    html_body: str,
    text_body: str | None = None,
) -> None:
    send_email_sync(
        to=to,
        subject=subject,
        html_body=html_body,
        text_body=text_body,
    )


@celery_app.task(
    name="backend.workers.tasks.index_rag_document_task",
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_jitter=True,
    max_retries=3,
)
def index_rag_document_task(*, document_id: str, user_id: str, job_id: str | None = None) -> None:
    index_document_sync(document_id=document_id, user_id=user_id, job_id=job_id)


@celery_app.task(
    name="backend.workers.tasks.cleanup_rag_document_task",
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_jitter=True,
    max_retries=5,
)
def cleanup_rag_document_task(
    *,
    document_id: str,
    user_id: str,
    storage_path: str | None,
) -> None:
    cleanup_document_sync(
        document_id=document_id,
        user_id=user_id,
        storage_path=storage_path,
    )


@celery_app.task(name="backend.workers.tasks.run_ai_evaluation_task")
def run_ai_evaluation_task(
    *,
    evaluation_run_id: str,
    user_id: str,
    dataset_id: str,
    prompt_version_id: str,
) -> None:
    run_evaluation_sync(
        evaluation_run_id=evaluation_run_id,
        user_id=user_id,
        dataset_id=dataset_id,
        prompt_version_id=prompt_version_id,
    )


@celery_app.task(
    name="backend.workers.tasks.extract_turn_memories_task",
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_jitter=True,
    max_retries=3,
)
def extract_turn_memories_task(
    *,
    user_id: str,
    agent_id: str,
    run_id: str,
    project_id: str | None,
    user_message: str,
    assistant_message: str,
    source_message_id: str,
) -> None:
    extract_turn_memories_sync(
        user_id=user_id,
        agent_id=agent_id,
        run_id=run_id,
        project_id=project_id,
        user_message=user_message,
        assistant_message=assistant_message,
        source_message_id=source_message_id,
    )


@celery_app.task(
    name="backend.workers.tasks.research_segmentation_task",
    **_research_task_options("research_light"),
)
def research_segmentation_task(self, *, run_id: str, user_id: str) -> None:
    segmentation_sync(run_id=run_id, user_id=user_id)


@celery_app.task(
    name="backend.workers.tasks.research_classifier_training_task",
    **_research_task_options("research_cpu"),
)
def research_classifier_training_task(self, *, run_id: str, user_id: str) -> None:
    classifier_training_sync(run_id=run_id, user_id=user_id)


@celery_app.task(
    name="backend.workers.tasks.research_topic_model_training_task",
    **_research_task_options("research_gpu"),
)
def research_topic_model_training_task(self, *, run_id: str, user_id: str) -> None:
    topic_model_training_sync(run_id=run_id, user_id=user_id)


@celery_app.task(
    name="backend.workers.tasks.research_topic_k_sweep_task",
    **_research_task_options("research_gpu"),
)
def research_topic_k_sweep_task(self, *, run_id: str, user_id: str) -> None:
    topic_k_sweep_sync(run_id=run_id, user_id=user_id)


@celery_app.task(
    name="backend.workers.tasks.research_topic_seed_stability_task",
    **_research_task_options("research_gpu"),
)
def research_topic_seed_stability_task(self, *, run_id: str, user_id: str) -> None:
    topic_seed_stability_sync(run_id=run_id, user_id=user_id)


@celery_app.task(
    name="backend.workers.tasks.research_robustness_sweep_task",
    **_research_task_options("research_cpu"),
)
def research_robustness_sweep_task(self, *, run_id: str, user_id: str) -> None:
    robustness_sweep_sync(run_id=run_id, user_id=user_id)


@celery_app.task(
    name="backend.workers.tasks.research_prediction_task",
    **_research_task_options("research_cpu"),
)
def research_prediction_task(self, *, run_id: str, user_id: str) -> None:
    prediction_sync(run_id=run_id, user_id=user_id)
