from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps.auth import get_current_user
from backend.api.deps.db import get_db
from backend.core.pagination import (
    PaginatedResponse,
    PaginationParams,
    paginated_response,
    pagination_params,
)
from backend.modules.ai.evaluation_service import AiEvaluationService
from backend.modules.ai.schemas import (
    AiEvaluationCaseCreate,
    AiEvaluationCaseResponse,
    AiEvaluationDatasetCreate,
    AiEvaluationDatasetResponse,
    AiEvaluationDatasetUpdate,
    AiEvaluationRunRequest,
    AiEvaluationRunResponse,
)
from backend.modules.ai.serializers import _dataset_case_to_response, _dataset_to_response
from backend.modules.identity_access.models import User

router = APIRouter()


@router.get("/evaluation-datasets", response_model=PaginatedResponse[AiEvaluationDatasetResponse])
async def list_datasets(
    pagination: PaginationParams = Depends(pagination_params),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = AiEvaluationService(db)
    datasets, total = await service.list_datasets(
        current_user, limit=pagination.limit, offset=pagination.offset
    )
    return paginated_response(
        [_dataset_to_response(item) for item in datasets],
        total=total,
        limit=pagination.limit,
        offset=pagination.offset,
    )


@router.post("/evaluation-datasets", response_model=AiEvaluationDatasetResponse, status_code=201)
async def create_dataset(
    payload: AiEvaluationDatasetCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = AiEvaluationService(db)
    dataset = await service.create_dataset(current_user, payload.name, payload.description)
    return _dataset_to_response(dataset)


@router.patch("/evaluation-datasets/{dataset_id}", response_model=AiEvaluationDatasetResponse)
async def update_dataset(
    dataset_id: str,
    payload: AiEvaluationDatasetUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = AiEvaluationService(db)
    dataset = await service.update_dataset(
        current_user, dataset_id, payload.model_dump(exclude_unset=True)
    )
    return _dataset_to_response(dataset)


@router.get(
    "/evaluation-datasets/{dataset_id}/cases",
    response_model=PaginatedResponse[AiEvaluationCaseResponse],
)
async def list_dataset_cases(
    dataset_id: str,
    pagination: PaginationParams = Depends(pagination_params),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = AiEvaluationService(db)
    cases, total = await service.list_dataset_cases(
        current_user,
        dataset_id,
        limit=pagination.limit,
        offset=pagination.offset,
    )
    return paginated_response(
        [_dataset_case_to_response(item) for item in cases],
        total=total,
        limit=pagination.limit,
        offset=pagination.offset,
    )


@router.post(
    "/evaluation-datasets/{dataset_id}/cases",
    response_model=AiEvaluationCaseResponse,
    status_code=201,
)
async def create_dataset_case(
    dataset_id: str,
    payload: AiEvaluationCaseCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = AiEvaluationService(db)
    case = await service.create_dataset_case(current_user, dataset_id, payload.model_dump())
    return _dataset_case_to_response(case)


@router.get("/evaluation-runs", response_model=PaginatedResponse[AiEvaluationRunResponse])
async def list_evaluation_runs(
    pagination: PaginationParams = Depends(pagination_params),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = AiEvaluationService(db)
    runs, total = await service.list_evaluation_runs(
        current_user, limit=pagination.limit, offset=pagination.offset
    )
    return paginated_response(
        [AiEvaluationRunResponse.model_validate(run) for run in runs],
        total=total,
        limit=pagination.limit,
        offset=pagination.offset,
    )


@router.post(
    "/evaluation-datasets/{dataset_id}/run",
    response_model=AiEvaluationRunResponse,
    status_code=202,
)
async def run_evaluation(
    dataset_id: str,
    payload: AiEvaluationRunRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = AiEvaluationService(db)
    evaluation_run = await service.queue_evaluation(
        current_user, dataset_id, payload.prompt_version_id
    )
    return AiEvaluationRunResponse.model_validate(evaluation_run)
