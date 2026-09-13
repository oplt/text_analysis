"""Contextual / mixed-method dataset routes (LATEST-017 split from ``routes.py``).

Included from ``routes.py`` without changing public URLs.
"""

from __future__ import annotations

import csv
import io

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps.auth import get_current_user
from backend.api.deps.db import get_db
from backend.modules.identity_access.models import User
from backend.modules.text_research.api.schemas import (
    ContextualDatasetCreate,
    ContextualDatasetDetail,
    ContextualDatasetSummary,
    ContextualImportResponse,
    ContextualLinkRequest,
    ContextualObservationPage,
)
from backend.modules.text_research.application.contextual_dataset_service import (
    ContextualDatasetService,
)

router = APIRouter()


@router.post(
    "/projects/{project_id}/contextual-datasets",
    response_model=ContextualDatasetSummary,
    status_code=201,
)
async def create_contextual_dataset(
    project_id: str,
    body: ContextualDatasetCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    dataset = await ContextualDatasetService(db).create_dataset(
        project_id=project_id,
        user_id=current_user.id,
        name=body.name,
        description=body.description,
    )
    return ContextualDatasetSummary(
        id=dataset.id,
        project_id=dataset.project_id,
        name=dataset.name,
        description=dataset.description,
        created_by=dataset.created_by,
        created_at=dataset.created_at,
        observation_count=0,
    )


@router.get(
    "/projects/{project_id}/contextual-datasets",
    response_model=list[ContextualDatasetSummary],
)
async def list_contextual_datasets(
    project_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    rows = await ContextualDatasetService(db).list_datasets(
        project_id=project_id, user_id=current_user.id
    )
    return [ContextualDatasetSummary.model_validate(row) for row in rows]


@router.get("/contextual-datasets/{dataset_id}", response_model=ContextualDatasetDetail)
async def get_contextual_dataset(
    dataset_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    detail = await ContextualDatasetService(db).get_dataset(dataset_id, user_id=current_user.id)
    return ContextualDatasetDetail.model_validate(detail)


@router.get(
    "/contextual-datasets/{dataset_id}/observations",
    response_model=ContextualObservationPage,
)
async def list_contextual_observations(
    dataset_id: str,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    page = await ContextualDatasetService(db).list_observations(
        dataset_id, user_id=current_user.id, limit=limit, offset=offset
    )
    return ContextualObservationPage.model_validate(page)


@router.delete("/contextual-datasets/{dataset_id}", status_code=204)
async def delete_contextual_dataset(
    dataset_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    await ContextualDatasetService(db).delete_dataset(dataset_id, user_id=current_user.id)


@router.post(
    "/contextual-datasets/{dataset_id}/import-csv",
    response_model=ContextualImportResponse,
)
async def import_contextual_csv(
    dataset_id: str,
    file: UploadFile = File(...),
    replace_existing: bool = Query(default=True),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        reader = csv.DictReader(io.TextIOWrapper(file.file, encoding="utf-8-sig", newline=""))
    except UnicodeError as exc:
        raise HTTPException(status_code=422, detail="CSV must be UTF-8 encoded.") from exc
    result = await ContextualDatasetService(db).import_csv(
        dataset_id,
        user_id=current_user.id,
        reader=reader,
        replace_existing=replace_existing,
    )
    return ContextualImportResponse.model_validate(result)


@router.post("/contextual-datasets/{dataset_id}/link-discourse")
async def link_contextual_discourse(
    dataset_id: str,
    body: ContextualLinkRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await ContextualDatasetService(db).link_discourse(
        dataset_id,
        user_id=current_user.id,
        corpus_id=body.corpus_id,
        codebook_id=body.codebook_id,
        label_ids=body.label_ids,
        indicator_key=body.indicator_key,
        unit_type=body.unit_type,
        group_by=body.group_by,
        join_on_year=body.join_on_year,
        provenance_mode=body.provenance_mode,
        model_id=body.model_id,
    )
