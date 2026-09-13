"""Export routes (LATEST-017 split from ``routes.py``).

Included from ``routes.py`` without changing public URLs.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps.auth import get_current_user
from backend.api.deps.db import get_db
from backend.modules.identity_access.models import User
from backend.modules.text_research.api.schemas import ExportManifestResponse, QuantedaScriptResponse
from backend.modules.text_research.application.export_service import ExportService

router = APIRouter()


@router.get("/runs/{run_id}/export.json")
async def export_run_json(
    run_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await ExportService(db).export_run_json(run_id, user_id=current_user.id)


@router.get("/codebooks/{codebook_id}/export.json")
async def export_codebook_json(
    codebook_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await ExportService(db).export_codebook_json(codebook_id, user_id=current_user.id)


@router.get("/classifiers/{model_id}/export/metrics.json")
async def export_model_metrics(
    model_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await ExportService(db).export_model_metrics(model_id, user_id=current_user.id)


@router.get("/preprocessing-profiles/{profile_id}/export.json")
async def export_preprocessing_profile_json(
    profile_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await ExportService(db).export_preprocessing_profile_json(
        profile_id, user_id=current_user.id
    )


@router.get("/corpora/{corpus_id}/export/manifest", response_model=ExportManifestResponse)
async def export_manifest(
    corpus_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    manifest = await ExportService(db).build_manifest(corpus_id, user_id=current_user.id)
    return ExportManifestResponse(manifest=manifest)


@router.get("/corpora/{corpus_id}/export/quanteda-script", response_model=QuantedaScriptResponse)
async def export_quanteda_script(
    corpus_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    script = await ExportService(db).build_quanteda_script(corpus_id, user_id=current_user.id)
    return QuantedaScriptResponse(script=script)


@router.get("/corpora/{corpus_id}/export/units.csv")
async def export_units_csv(
    corpus_id: str,
    unit_type: str = Query(default="paragraph"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return StreamingResponse(
        ExportService(db).iter_units_csv(corpus_id, user_id=current_user.id, unit_type=unit_type),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{corpus_id}-units.csv"'},
    )


@router.get("/corpora/{corpus_id}/export/annotations.csv")
async def export_annotations_csv(
    corpus_id: str,
    codebook_id: str = Query(...),
    max_rows: int | None = Query(default=None, ge=1),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = ExportService(db)
    meta = await service.annotation_export_meta(
        corpus_id,
        user_id=current_user.id,
        codebook_id=codebook_id,
        max_rows=max_rows,
    )
    headers = {
        "Content-Disposition": f'attachment; filename="{corpus_id}-annotations.csv"',
        **meta.as_headers(),
    }
    return StreamingResponse(
        service.iter_annotations_csv(
            corpus_id,
            user_id=current_user.id,
            codebook_id=codebook_id,
            max_rows=max_rows,
        ),
        media_type="text/csv",
        headers=headers,
    )


@router.get("/classifiers/{model_id}/export/predictions.csv")
async def export_predictions_csv(
    model_id: str,
    max_rows: int | None = Query(default=None, ge=1),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = ExportService(db)
    meta = await service.prediction_export_meta(
        model_id, user_id=current_user.id, max_rows=max_rows
    )
    headers = {
        "Content-Disposition": f'attachment; filename="{model_id}-predictions.csv"',
        **meta.as_headers(),
    }
    return StreamingResponse(
        service.iter_predictions_csv(model_id, user_id=current_user.id, max_rows=max_rows),
        media_type="text/csv",
        headers=headers,
    )
