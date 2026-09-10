from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from backend.core.schemas import RequestModel

AI_KEY_PATTERN = r"^[a-z0-9_][a-z0-9_\-\.]{1,127}$"


class AiProviderDescriptor(BaseModel):
    key: str
    label: str
    supports_generation: bool
    supports_embeddings: bool


class AiVariableDefinition(BaseModel):
    name: str = Field(min_length=1, max_length=64, pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$")
    description: str | None = None
    required: bool = True


class AiPromptTemplateCreate(RequestModel):
    key: str = Field(min_length=2, max_length=128, pattern=AI_KEY_PATTERN)
    name: str = Field(min_length=2, max_length=255)
    description: str | None = None


class AiPromptTemplateUpdate(RequestModel):
    name: str | None = Field(default=None, min_length=2, max_length=255)
    description: str | None = None
    is_active: bool | None = None
    active_version_id: str | None = None


class AiPromptTemplateResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    key: str
    name: str
    description: str | None
    is_active: bool
    active_version_id: str | None
    created_at: datetime
    updated_at: datetime


class AiPromptVersionCreate(RequestModel):
    provider_key: str = Field(min_length=2, max_length=64)
    model_name: str = Field(min_length=2, max_length=128)
    system_prompt: str = ""
    user_prompt_template: str = Field(min_length=1)
    variable_definitions: list[AiVariableDefinition] = Field(default_factory=list)
    response_format: Literal["text", "json"] = "text"
    temperature: float = Field(default=0.2, ge=0, le=2)
    rollout_percentage: int = Field(default=100, ge=0, le=100)
    is_published: bool = False
    input_cost_per_million: int = Field(default=0, ge=0)
    output_cost_per_million: int = Field(default=0, ge=0)


class AiPromptVersionUpdate(RequestModel):
    system_prompt: str | None = None
    user_prompt_template: str | None = Field(default=None, min_length=1)
    variable_definitions: list[AiVariableDefinition] | None = None
    response_format: Literal["text", "json"] | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    rollout_percentage: int | None = Field(default=None, ge=0, le=100)
    is_published: bool | None = None
    input_cost_per_million: int | None = Field(default=None, ge=0)
    output_cost_per_million: int | None = Field(default=None, ge=0)


class AiPromptVersionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    prompt_template_id: str
    version_number: int
    provider_key: str
    model_name: str
    system_prompt: str
    user_prompt_template: str
    variable_definitions: list[AiVariableDefinition]
    response_format: str
    temperature: float
    rollout_percentage: int
    is_published: bool
    input_cost_per_million: int
    output_cost_per_million: int
    created_by_user_id: str | None
    created_at: datetime


class AiDocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    description: str | None
    filename: str | None
    content_type: str
    size_bytes: int
    ingestion_status: str
    metadata: dict[str, Any]
    chunk_count: int
    created_at: datetime
    updated_at: datetime


class AiDocumentCreate(RequestModel):
    title: str = Field(min_length=2, max_length=255)
    description: str | None = None
    content: str = Field(min_length=1)
    content_type: str = Field(default="text/plain", min_length=3, max_length=128)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AiChunkMatchResponse(BaseModel):
    document_id: str
    chunk_id: str
    document_title: str
    chunk_index: int
    score: float
    content: str


class AiRetrieveRequest(RequestModel):
    query: str = Field(min_length=2)
    document_ids: list[str] = Field(default_factory=list)
    top_k: int = Field(default=5, ge=1, le=20)


class AiRunRequest(RequestModel):
    prompt_template_key: str | None = Field(default=None, min_length=2, max_length=128)
    prompt_version_id: str | None = None
    variables: dict[str, Any] = Field(default_factory=dict)
    retrieval_query: str | None = None
    document_ids: list[str] = Field(default_factory=list)
    top_k: int = Field(default=4, ge=1, le=20)
    review_required: bool = False


class AiRunResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    prompt_template_id: str | None
    prompt_version_id: str | None
    agent_id: str | None = None
    agent_run_id: str | None = None
    provider_key: str
    model_name: str
    status: str
    response_format: str
    variables: dict[str, Any]
    retrieval_query: str | None
    retrieved_chunk_ids: list[str]
    retrieval_degraded: bool = False
    memory_degraded: bool = False
    degradation_reason: str | None = None
    injection_chunks_filtered: int = 0
    input_messages: list[dict[str, Any]]
    output_text: str | None
    output_json: dict[str, Any] | None
    latency_ms: int | None
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost_micros: int
    error_message: str | None
    review_status: str
    created_at: datetime
    completed_at: datetime | None


class AiReviewCreate(RequestModel):
    assigned_to_user_id: str | None = None


class AiReviewDecision(RequestModel):
    status: Literal["approved", "rejected", "changes_requested"]
    reviewer_notes: str | None = None
    corrected_output: str | None = None


class AiReviewItemResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    run_id: str
    requested_by_user_id: str
    assigned_to_user_id: str | None
    reviewed_by_user_id: str | None
    status: str
    reviewer_notes: str | None
    corrected_output: str | None
    created_at: datetime
    updated_at: datetime


class AiFeedbackCreate(RequestModel):
    rating: Literal[-1, 1]
    comment: str | None = None
    corrected_output: str | None = None


class AiFeedbackResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    run_id: str
    user_id: str
    rating: int
    comment: str | None
    corrected_output: str | None
    created_at: datetime


class AiEvaluationDatasetCreate(RequestModel):
    name: str = Field(min_length=2, max_length=255)
    description: str | None = None


class AiEvaluationDatasetUpdate(RequestModel):
    name: str | None = Field(default=None, min_length=2, max_length=255)
    description: str | None = None


class AiEvaluationDatasetResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    description: str | None
    created_at: datetime
    updated_at: datetime


class AiEvaluationCaseCreate(RequestModel):
    input_variables: dict[str, Any] = Field(default_factory=dict)
    retrieval_query: str | None = Field(default=None, min_length=1, max_length=4000)
    document_ids: list[str] = Field(default_factory=list)
    expected_chunk_ids: list[str] = Field(default_factory=list)
    expected_output_text: str | None = None
    expected_output_json: dict[str, Any] | None = None
    notes: str | None = None


class AiEvaluationCaseResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    dataset_id: str
    input_variables: dict[str, Any]
    retrieval_query: str | None
    document_ids: list[str]
    expected_chunk_ids: list[str]
    expected_output_text: str | None
    expected_output_json: dict[str, Any] | None
    notes: str | None
    created_at: datetime


class AiEvaluationRunRequest(RequestModel):
    prompt_version_id: str


class AiEvaluationRunItemResponse(BaseModel):
    evaluation_case_id: str
    ai_run_id: str
    score: float
    passed: bool
    notes: str | None


class AiEvaluationRunResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    dataset_id: str
    prompt_version_id: str
    status: str
    total_cases: int
    passed_cases: int
    average_score: float
    created_at: datetime
    completed_at: datetime | None
    items: list[AiEvaluationRunItemResponse] = Field(default_factory=list)


class AiModuleOverviewResponse(BaseModel):
    providers: list[AiProviderDescriptor]
    prompt_templates: list[AiPromptTemplateResponse]
    recent_runs: list[AiRunResponse]
    documents: list[AiDocumentResponse]
    datasets: list[AiEvaluationDatasetResponse]
