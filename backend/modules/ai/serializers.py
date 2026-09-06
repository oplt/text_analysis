from __future__ import annotations

from backend.modules.ai.schemas import (
    AiDocumentResponse,
    AiEvaluationCaseResponse,
    AiEvaluationDatasetResponse,
    AiFeedbackResponse,
    AiPromptTemplateResponse,
    AiPromptVersionResponse,
    AiReviewItemResponse,
    AiRunResponse,
)


def run_to_response(run) -> AiRunResponse:
    return AiRunResponse(
        id=run.id,
        prompt_template_id=run.prompt_template_id,
        prompt_version_id=run.prompt_version_id,
        provider_key=run.provider_key,
        model_name=run.model_name,
        status=run.status,
        response_format=run.response_format,
        variables=run.variables_json,
        retrieval_query=run.retrieval_query,
        retrieved_chunk_ids=run.retrieved_chunk_ids_json,
        retrieval_degraded=getattr(run, "retrieval_degraded", False),
        memory_degraded=getattr(run, "memory_degraded", False),
        degradation_reason=getattr(run, "degradation_reason", None),
        injection_chunks_filtered=getattr(run, "injection_chunks_filtered", 0),
        input_messages=run.input_messages_json,
        output_text=run.output_text,
        output_json=run.output_json,
        latency_ms=run.latency_ms,
        input_tokens=run.input_tokens,
        output_tokens=run.output_tokens,
        total_tokens=run.total_tokens,
        estimated_cost_micros=run.estimated_cost_micros,
        error_message=run.error_message,
        review_status=run.review_status,
        created_at=run.created_at,
        completed_at=run.completed_at,
    )


def _prompt_template_to_response(template) -> AiPromptTemplateResponse:
    return AiPromptTemplateResponse.model_validate(template)


def _prompt_version_to_response(version) -> AiPromptVersionResponse:
    return AiPromptVersionResponse(
        id=version.id,
        prompt_template_id=version.prompt_template_id,
        version_number=version.version_number,
        provider_key=version.provider_key,
        model_name=version.model_name,
        system_prompt=version.system_prompt,
        user_prompt_template=version.user_prompt_template,
        variable_definitions=version.variable_definitions_json,
        response_format=version.response_format,
        temperature=version.temperature,
        rollout_percentage=version.rollout_percentage,
        is_published=version.is_published,
        input_cost_per_million=version.input_cost_per_million,
        output_cost_per_million=version.output_cost_per_million,
        created_by_user_id=version.created_by_user_id,
        created_at=version.created_at,
    )


def _document_to_response(document) -> AiDocumentResponse:
    return AiDocumentResponse(
        id=document.id,
        title=document.title,
        description=document.description,
        filename=document.filename,
        content_type=document.content_type,
        size_bytes=document.size_bytes,
        ingestion_status=document.ingestion_status,
        metadata=document.metadata_json,
        chunk_count=document.chunk_count,
        created_at=document.created_at,
        updated_at=document.updated_at,
    )


def _review_to_response(review) -> AiReviewItemResponse:
    return AiReviewItemResponse.model_validate(review)


def _feedback_to_response(feedback) -> AiFeedbackResponse:
    return AiFeedbackResponse.model_validate(feedback)


def _dataset_to_response(dataset) -> AiEvaluationDatasetResponse:
    return AiEvaluationDatasetResponse.model_validate(dataset)


def _dataset_case_to_response(case) -> AiEvaluationCaseResponse:
    return AiEvaluationCaseResponse(
        id=case.id,
        dataset_id=case.dataset_id,
        input_variables=case.input_variables_json,
        retrieval_query=case.retrieval_query,
        document_ids=case.document_ids_json,
        expected_chunk_ids=case.expected_chunk_ids_json,
        expected_output_text=case.expected_output_text,
        expected_output_json=case.expected_output_json,
        notes=case.notes,
        created_at=case.created_at,
    )
