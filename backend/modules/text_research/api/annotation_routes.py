"""Codebooks/labels, annotation, and reliability/adjudication routes
(LATEST-017 split from ``routes.py``).

Included from ``routes.py`` without changing public URLs.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps.auth import get_current_user
from backend.api.deps.db import get_db
from backend.core.pagination import (
    PaginatedResponse,
    PaginationParams,
    paginated_response,
    pagination_params,
)
from backend.modules.identity_access.models import User
from backend.modules.text_research.api.route_serializers import _label_response, _run_response
from backend.modules.text_research.api.schemas import (
    AdjudicationSaveRequest,
    AnalysisRunResponse,
    AnnotationAssignRequest,
    AnnotationCampaignAssignRequest,
    AnnotationCampaignCreate,
    AnnotationCampaignResponse,
    AnnotationCampaignUpdate,
    AnnotationLabelCreate,
    AnnotationLabelResponse,
    AnnotationLabelUpdate,
    AnnotationResponse,
    AnnotationSaveRequest,
    CodebookCreate,
    CodebookResponse,
    CorpusAnnotationAssignRequest,
    CorpusAnnotationAssignResponse,
    ReliabilityRequest,
    TextUnitContextResponse,
)
from backend.modules.text_research.application.adjudication_service import AdjudicationService
from backend.modules.text_research.application.annotation_service import AnnotationService
from backend.modules.text_research.application.campaign_service import AnnotationCampaignService
from backend.modules.text_research.application.codebook_service import CodebookService
from backend.modules.text_research.application.reliability_service import ReliabilityService

router = APIRouter()


# ------------------------------------------------------------------
# Codebooks & labels
# ------------------------------------------------------------------


@router.post("/projects/{project_id}/codebooks", response_model=CodebookResponse, status_code=201)
async def create_codebook(
    project_id: str,
    body: CodebookCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    codebook = await CodebookService(db).create_codebook(
        project_id=project_id,
        user_id=current_user.id,
        name=body.name,
        description=body.description,
        seed_placeholder_labels=body.seed_demo_labels,
    )
    return CodebookResponse.model_validate(codebook)


@router.get("/projects/{project_id}/codebooks", response_model=list[CodebookResponse])
async def list_codebooks(
    project_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    codebooks = await CodebookService(db).list_codebooks(
        project_id=project_id, user_id=current_user.id
    )
    return [CodebookResponse.model_validate(c) for c in codebooks]


@router.get("/codebooks/{codebook_id}", response_model=CodebookResponse)
async def get_codebook(
    codebook_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    codebook = await CodebookService(db).get_codebook(codebook_id, user_id=current_user.id)
    return CodebookResponse.model_validate(codebook)


@router.post("/codebooks/{codebook_id}/versions", response_model=CodebookResponse, status_code=201)
async def create_codebook_version(
    codebook_id: str,
    new_version: str = Query(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    codebook = await CodebookService(db).create_version(
        codebook_id, user_id=current_user.id, new_version=new_version
    )
    return CodebookResponse.model_validate(codebook)


@router.post("/codebooks/{codebook_id}/freeze", response_model=CodebookResponse)
async def freeze_codebook(
    codebook_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    codebook = await CodebookService(db).freeze_codebook(codebook_id, user_id=current_user.id)
    return CodebookResponse.model_validate(codebook)


@router.post(
    "/codebooks/{codebook_id}/labels", response_model=AnnotationLabelResponse, status_code=201
)
async def add_label(
    codebook_id: str,
    body: AnnotationLabelCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    label = await CodebookService(db).add_label(
        codebook_id,
        user_id=current_user.id,
        name=body.name,
        description=body.description,
        inclusion_criteria=body.inclusion_criteria,
        exclusion_criteria=body.exclusion_criteria,
        positive_examples=body.positive_examples,
        negative_examples=body.negative_examples,
    )
    return _label_response(label)


@router.get("/codebooks/{codebook_id}/labels", response_model=list[AnnotationLabelResponse])
async def list_labels(
    codebook_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    labels = await CodebookService(db).list_labels(codebook_id, user_id=current_user.id)
    return [_label_response(label) for label in labels]


@router.patch("/labels/{label_id}", response_model=AnnotationLabelResponse)
async def update_label(
    label_id: str,
    body: AnnotationLabelUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    fields = body.model_dump(exclude_unset=True)
    label = await CodebookService(db).update_label(label_id, user_id=current_user.id, **fields)
    return _label_response(label)


# ------------------------------------------------------------------
# Annotation
# ------------------------------------------------------------------


@router.post("/annotations/assign", status_code=201)
async def assign_annotation_tasks(
    body: AnnotationAssignRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    tasks = await AnnotationService(db).assign_tasks(
        user_id=current_user.id,
        text_unit_ids=body.text_unit_ids,
        annotator_ids=body.annotator_ids,
    )
    return {"assigned_count": len(tasks)}


@router.post(
    "/corpora/{corpus_id}/annotations/assign",
    response_model=CorpusAnnotationAssignResponse,
    status_code=201,
)
async def assign_corpus_annotation_tasks(
    corpus_id: str,
    body: CorpusAnnotationAssignRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    sample_size = body.sample_size if body.limit is None else body.limit
    return await AnnotationService(db).assign_corpus_tasks(
        corpus_id,
        user_id=current_user.id,
        unit_type=body.unit_type,
        annotator_ids=body.annotator_ids,
        sample_size=sample_size,
        strategy=body.strategy,
        overlap_count=body.overlap_count,
        overlap_percent=body.overlap_percent,
        random_seed=body.random_seed,
        stratify_by=body.stratify_by,
        stratum_mode=body.stratum_mode,
        sampling_level=body.sampling_level,
        max_units_per_document=body.max_units_per_document,
        campaign_id=body.campaign_id,
        create_campaign=body.create_campaign and body.campaign_id is None,
        campaign_name=body.campaign_name,
        campaign_description=body.campaign_description,
        annotation_mode=body.annotation_mode,
        blind_mode=body.blind_mode,
        ai_assistance_enabled=body.ai_assistance_enabled,
        reveal_after=body.reveal_after,
        codebook_id=body.codebook_id,
    )


@router.post(
    "/projects/{project_id}/annotation-campaigns",
    response_model=AnnotationCampaignResponse,
    status_code=201,
)
async def create_annotation_campaign(
    project_id: str,
    body: AnnotationCampaignCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = AnnotationCampaignService(db)
    campaign = await service.create_campaign_committed(
        user_id=current_user.id,
        project_id=project_id,
        corpus_id=body.corpus_id,
        name=body.name,
        description=body.description,
        codebook_id=body.codebook_id,
        unit_type=body.unit_type,
        sampling_strategy=body.sampling_strategy,
        assignment_strategy=body.assignment_strategy,
        sample_size=body.sample_size,
        overlap_count=body.overlap_count,
        overlap_percent=body.overlap_percent,
        annotation_mode=body.annotation_mode,
        blind_mode=body.blind_mode,
        ai_assistance_enabled=body.ai_assistance_enabled,
        reveal_after=body.reveal_after,
        annotator_ids=body.annotator_ids,
        metadata=body.metadata,
    )
    return await service.campaign_payload(campaign)


@router.get(
    "/projects/{project_id}/annotation-campaigns",
    response_model=list[AnnotationCampaignResponse],
)
async def list_annotation_campaigns(
    project_id: str,
    corpus_id: str | None = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = AnnotationCampaignService(db)
    campaigns = await service.list_campaigns(
        project_id, user_id=current_user.id, corpus_id=corpus_id
    )
    return [await service.campaign_payload(campaign) for campaign in campaigns]


@router.get(
    "/annotation-campaigns/{campaign_id}",
    response_model=AnnotationCampaignResponse,
)
async def get_annotation_campaign(
    campaign_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = AnnotationCampaignService(db)
    campaign = await service.get_campaign(campaign_id, user_id=current_user.id)
    return await service.campaign_payload(campaign)


@router.patch(
    "/annotation-campaigns/{campaign_id}",
    response_model=AnnotationCampaignResponse,
)
async def patch_annotation_campaign(
    campaign_id: str,
    body: AnnotationCampaignUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = AnnotationCampaignService(db)
    campaign = await service.patch_campaign(
        campaign_id,
        user_id=current_user.id,
        name=body.name,
        description=body.description,
        status=body.status,
        annotation_mode=body.annotation_mode,
        blind_mode=body.blind_mode,
        ai_assistance_enabled=body.ai_assistance_enabled,
        reveal_after=body.reveal_after,
        metadata=body.metadata,
    )
    return await service.campaign_payload(campaign)


@router.post("/annotation-campaigns/{campaign_id}/assign", status_code=201)
async def assign_annotation_campaign(
    campaign_id: str,
    body: AnnotationCampaignAssignRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await AnnotationCampaignService(db).assign(
        campaign_id,
        user_id=current_user.id,
        annotator_ids=body.annotator_ids,
        sample_size=body.sample_size,
        strategy=body.strategy,
        overlap_count=body.overlap_count,
        overlap_percent=body.overlap_percent,
        random_seed=body.random_seed,
        stratify_by=body.stratify_by,
        stratum_mode=body.stratum_mode,
        sampling_level=body.sampling_level,
        max_units_per_document=body.max_units_per_document,
    )


@router.get("/annotation-campaigns/{campaign_id}/progress")
async def annotation_campaign_progress(
    campaign_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await AnnotationCampaignService(db).progress(campaign_id, user_id=current_user.id)


@router.get("/text-units/{text_unit_id}/blind-policy")
async def text_unit_blind_policy(
    text_unit_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    await AnnotationService(db).get_text_unit_or_404(text_unit_id, user_id=current_user.id)
    return await AnnotationCampaignService(db).blind_policy_for_annotator_unit(
        text_unit_id=text_unit_id, annotator_id=current_user.id
    )


@router.get("/annotations/queue", response_model=PaginatedResponse[dict[str, Any]])
async def list_annotation_queue(
    status: str | None = None,
    pagination: PaginationParams = Depends(pagination_params),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    queue, total = await AnnotationService(db).list_queue(
        requesting_user_id=current_user.id,
        status=status,
        limit=pagination.limit,
        offset=pagination.offset,
    )
    return paginated_response(queue, total=total, limit=pagination.limit, offset=pagination.offset)


@router.post("/annotations", response_model=list[AnnotationResponse])
async def save_annotations(
    body: AnnotationSaveRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    annotations = await AnnotationService(db).save_annotations(
        user_id=current_user.id,
        text_unit_id=body.text_unit_id,
        codebook_id=body.codebook_id,
        values=body.values,
        campaign_id=body.campaign_id,
        mark_task_complete=body.mark_task_complete,
    )
    return [AnnotationResponse.model_validate(a) for a in annotations]


@router.get("/corpora/{corpus_id}/annotations/progress")
async def annotation_progress(
    corpus_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await AnnotationService(db).progress(corpus_id, user_id=current_user.id)


@router.get("/text-units/{text_unit_id}/annotations")
async def list_unit_annotations(
    text_unit_id: str,
    campaign_id: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    annotations = await AnnotationService(db).list_annotations_for_unit(
        text_unit_id, user_id=current_user.id, campaign_id=campaign_id
    )
    return [AnnotationResponse.model_validate(a) for a in annotations]


@router.get(
    "/corpora/{corpus_id}/annotations",
    response_model=list[AnnotationResponse],
)
async def list_corpus_annotations_for_units(
    corpus_id: str,
    text_unit_ids: list[str] = Query(default=[]),
    campaign_id: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List human annotations for specific units.

    Predictions and adjudications are never returned here — callers must keep
    MODEL PREDICTION / HUMAN ANNOTATION / ADJUDICATED GOLD layers separate.
    """
    annotations = await AnnotationService(db).list_annotations_for_units(
        corpus_id,
        user_id=current_user.id,
        text_unit_ids=text_unit_ids,
        campaign_id=campaign_id,
    )
    return [AnnotationResponse.model_validate(a) for a in annotations]


@router.get("/text-units/{text_unit_id}/context", response_model=TextUnitContextResponse)
async def get_text_unit_context(
    text_unit_id: str,
    window: int = Query(default=2, ge=0, le=10),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await AnnotationService(db).get_unit_context(
        text_unit_id, user_id=current_user.id, window=window
    )


# ------------------------------------------------------------------
# Reliability & adjudication
# ------------------------------------------------------------------


@router.post("/corpora/{corpus_id}/reliability", response_model=AnalysisRunResponse)
async def compute_reliability(
    corpus_id: str,
    body: ReliabilityRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    run = await ReliabilityService(db).compute_reliability(
        corpus_id,
        user_id=current_user.id,
        codebook_id=body.codebook_id,
        label_ids=body.label_ids,
        campaign_id=body.campaign_id,
        unit_type=body.unit_type,
        annotator_ids=body.annotator_ids,
        bootstrap_samples=body.bootstrap_samples,
        confidence_level=body.confidence_level,
        random_seed=body.random_seed,
    )
    return _run_response(run)


@router.post(
    "/annotation-campaigns/{campaign_id}/reliability",
    response_model=AnalysisRunResponse,
)
async def compute_campaign_reliability(
    campaign_id: str,
    body: ReliabilityRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    campaign = await AnnotationCampaignService(db).get_campaign(
        campaign_id, user_id=current_user.id
    )
    codebook_id = body.codebook_id or campaign.codebook_id
    if not codebook_id:
        raise HTTPException(
            status_code=422, detail="codebook_id is required when the campaign has none"
        )
    run = await ReliabilityService(db).compute_reliability(
        campaign.corpus_id,
        user_id=current_user.id,
        codebook_id=codebook_id,
        label_ids=body.label_ids,
        campaign_id=campaign_id,
        unit_type=body.unit_type or campaign.unit_type,
        annotator_ids=body.annotator_ids,
        bootstrap_samples=body.bootstrap_samples,
        confidence_level=body.confidence_level,
        random_seed=body.random_seed,
    )
    return _run_response(run)


@router.get("/corpora/{corpus_id}/adjudication/disagreements")
async def list_disagreements(
    corpus_id: str,
    codebook_id: str = Query(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await AdjudicationService(db).list_disagreements(
        corpus_id, user_id=current_user.id, codebook_id=codebook_id
    )


@router.post("/adjudication", status_code=201)
async def save_adjudication(
    body: AdjudicationSaveRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    adjudication = await AdjudicationService(db).save_adjudication(
        user_id=current_user.id,
        text_unit_id=body.text_unit_id,
        label_id=body.label_id,
        final_value=body.final_value,
        comment=body.comment,
        campaign_id=body.campaign_id,
    )
    return {"id": adjudication.id}


@router.get("/corpora/{corpus_id}/adjudications")
async def list_adjudications(
    corpus_id: str,
    campaign_id: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await AdjudicationService(db).list_adjudications(
        corpus_id, user_id=current_user.id, campaign_id=campaign_id
    )


@router.get("/annotation-campaigns/{campaign_id}/disagreements")
async def list_campaign_disagreements(
    campaign_id: str,
    codebook_id: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    campaign = await AnnotationCampaignService(db).get_campaign(
        campaign_id, user_id=current_user.id
    )
    resolved_codebook_id = codebook_id or campaign.codebook_id
    if not resolved_codebook_id:
        raise HTTPException(status_code=422, detail="Campaign has no codebook")
    return await AdjudicationService(db).list_disagreements(
        campaign.corpus_id,
        user_id=current_user.id,
        codebook_id=resolved_codebook_id,
        campaign_id=campaign_id,
    )


@router.post("/annotation-campaigns/{campaign_id}/adjudications", status_code=201)
async def save_campaign_adjudication(
    campaign_id: str,
    body: AdjudicationSaveRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    adjudication = await AdjudicationService(db).save_adjudication(
        user_id=current_user.id,
        text_unit_id=body.text_unit_id,
        label_id=body.label_id,
        final_value=body.final_value,
        comment=body.comment,
        campaign_id=campaign_id,
    )
    return {"id": adjudication.id}


@router.get("/annotation-campaigns/{campaign_id}/adjudications")
async def list_campaign_adjudications(
    campaign_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    campaign = await AnnotationCampaignService(db).get_campaign(
        campaign_id, user_id=current_user.id
    )
    return await AdjudicationService(db).list_adjudications(
        campaign.corpus_id, user_id=current_user.id, campaign_id=campaign_id
    )
