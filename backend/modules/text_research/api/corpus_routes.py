"""Corpus lifecycle routes: demo seed, corpora, documents, cleaning, segmentation,
preprocessing profiles (LATEST-017 split from ``routes.py``).

Included from ``routes.py`` without changing public URLs.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Query, UploadFile
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
from backend.modules.text_research.api.route_serializers import (
    _cleaning_profile_response,
    _corpus_response,
    _document_response,
    _loads,
    _profile_response,
    _run_response,
)
from backend.modules.text_research.api.schemas import (
    AnalysisRunResponse,
    ApplyCleaningRequest,
    BulkMetadataUpdate,
    CleaningPreviewRequest,
    CleaningPreviewResponse,
    CleaningProfileCreate,
    CleaningProfileResponse,
    CleaningProfileUpdate,
    CorpusDocumentCreate,
    CorpusDocumentResponse,
    CorpusDocumentUpdate,
    DemoSeedRequest,
    MetadataImportResponse,
    PreprocessingPreviewRequest,
    PreprocessingPreviewResponse,
    PreprocessingProfileCreate,
    PreprocessingProfileResponse,
    PreprocessingProfileUpdate,
    ResearchCorpusCreate,
    ResearchCorpusResponse,
    ResearchCorpusUpdate,
    SegmentRequest,
    SourceTextResponse,
)
from backend.modules.text_research.application.cleaning_service import CleaningProfileService
from backend.modules.text_research.application.corpus_service import CorpusService
from backend.modules.text_research.application.demo_seed_service import DemoSeedService
from backend.modules.text_research.application.ingestion_qa_service import IngestionQaService
from backend.modules.text_research.application.preprocessing_service import (
    PreprocessingProfileService,
)
from backend.modules.text_research.application.segmentation_service import SegmentationService

router = APIRouter()


# ------------------------------------------------------------------
# Demo seed
# ------------------------------------------------------------------


@router.post(
    "/projects/{project_id}/demo-seed", response_model=ResearchCorpusResponse, status_code=201
)
async def seed_demo_corpus(
    project_id: str,
    body: DemoSeedRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    corpus = await DemoSeedService(db).seed_demo_corpus(
        project_id=project_id,
        user_id=current_user.id,
        corpus_name=body.corpus_name or "Demo Corpus (Synthetic)",
    )
    return _corpus_response(corpus)


# ------------------------------------------------------------------
# Corpora
# ------------------------------------------------------------------


@router.post(
    "/projects/{project_id}/corpora", response_model=ResearchCorpusResponse, status_code=201
)
async def create_corpus(
    project_id: str,
    body: ResearchCorpusCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    corpus = await CorpusService(db).create_corpus(
        project_id=project_id,
        user_id=current_user.id,
        name=body.name,
        description=body.description,
    )
    return _corpus_response(corpus)


@router.get("/projects/{project_id}/corpora", response_model=list[ResearchCorpusResponse])
async def list_corpora(
    project_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    corpora = await CorpusService(db).list_corpora(project_id=project_id, user_id=current_user.id)
    return [_corpus_response(c) for c in corpora]


@router.get("/corpora/{corpus_id}", response_model=ResearchCorpusResponse)
async def get_corpus(
    corpus_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    corpus = await CorpusService(db).get_corpus(corpus_id, user_id=current_user.id)
    return _corpus_response(corpus)


@router.patch("/corpora/{corpus_id}", response_model=ResearchCorpusResponse)
async def update_corpus(
    corpus_id: str,
    body: ResearchCorpusUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    corpus = await CorpusService(db).update_corpus(
        corpus_id, user_id=current_user.id, name=body.name, description=body.description
    )
    return _corpus_response(corpus)


@router.delete("/corpora/{corpus_id}", status_code=204)
async def delete_corpus(
    corpus_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    await CorpusService(db).delete_corpus(corpus_id, user_id=current_user.id)


# ------------------------------------------------------------------
# Corpus documents
# ------------------------------------------------------------------


@router.post(
    "/corpora/{corpus_id}/documents", response_model=CorpusDocumentResponse, status_code=201
)
async def add_document(
    corpus_id: str,
    body: CorpusDocumentCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    document = await CorpusService(db).add_document(
        corpus_id,
        user_id=current_user.id,
        rag_document_id=body.rag_document_id,
        title=body.title,
        organization=body.organization,
        organization_type=body.organization_type,
        publication_year=body.publication_year,
        publication_type=body.publication_type,
        country=body.country,
        region=body.region,
        cultural_sphere=body.cultural_sphere,
        language=body.language,
        education_level=body.education_level,
        source_url=body.source_url,
        research_notes=body.research_notes,
        extra_metadata=body.metadata_json,
    )
    return _document_response(document)


@router.get(
    "/corpora/{corpus_id}/documents", response_model=PaginatedResponse[CorpusDocumentResponse]
)
async def list_documents(
    corpus_id: str,
    pagination: PaginationParams = Depends(pagination_params),
    organization: str | None = Query(default=None),
    publication_year: int | None = Query(default=None),
    region: str | None = Query(default=None),
    cultural_sphere: str | None = Query(default=None),
    language: str | None = Query(default=None),
    search: str | None = Query(default=None),
    sort_by: str = Query(default="created_at"),
    sort_dir: str = Query(default="asc"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    documents, total = await CorpusService(db).paginate_documents(
        corpus_id,
        user_id=current_user.id,
        organization=organization,
        publication_year=publication_year,
        region=region,
        cultural_sphere=cultural_sphere,
        language=language,
        search=search,
        sort_by=sort_by,
        sort_dir=sort_dir,
        limit=pagination.limit,
        offset=pagination.offset,
    )
    return paginated_response(
        [_document_response(d) for d in documents],
        total=total,
        limit=pagination.limit,
        offset=pagination.offset,
    )


@router.get("/documents/{document_id}", response_model=CorpusDocumentResponse)
async def get_document(
    document_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    document = await CorpusService(db).get_document(document_id, user_id=current_user.id)
    return _document_response(document)


@router.patch("/documents/{document_id}", response_model=CorpusDocumentResponse)
async def update_document(
    document_id: str,
    body: CorpusDocumentUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    fields = body.model_dump(exclude_unset=True)
    if "metadata_json" in fields and fields["metadata_json"] is not None:
        fields["extra_metadata"] = fields.pop("metadata_json")
    document = await CorpusService(db).update_document_metadata(
        document_id, user_id=current_user.id, **fields
    )
    return _document_response(document)


@router.post(
    "/corpora/{corpus_id}/documents/metadata/bulk", response_model=list[CorpusDocumentResponse]
)
async def bulk_update_metadata(
    corpus_id: str,
    body: BulkMetadataUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    documents = await CorpusService(db).bulk_update_metadata(
        corpus_id, user_id=current_user.id, document_ids=body.document_ids, fields=body.fields
    )
    return [_document_response(d) for d in documents]


@router.post(
    "/corpora/{corpus_id}/documents/metadata/import", response_model=MetadataImportResponse
)
async def import_metadata_csv(
    corpus_id: str,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    content = (await file.read()).decode("utf-8")
    result = await CorpusService(db).import_metadata_csv(
        corpus_id, user_id=current_user.id, csv_content=content
    )
    return MetadataImportResponse(**result)


@router.delete("/documents/{document_id}", status_code=204)
async def delete_document(
    document_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    await CorpusService(db).delete_document(document_id, user_id=current_user.id)


@router.get("/documents/{document_id}/source-text", response_model=SourceTextResponse)
async def get_source_text(
    document_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = CorpusService(db)
    source = await service.get_canonical_source(document_id, user_id=current_user.id)
    return SourceTextResponse(
        document_id=document_id,
        text=source.canonical_text,
        canonical_text_checksum=source.canonical_text_checksum,
        parser_name=source.parser_name,
        parser_version=source.parser_version,
        page_provenance=_loads(source.page_provenance_json, []),
        source="canonical",
    )


@router.post("/corpora/{corpus_id}/ingestion-qa", response_model=AnalysisRunResponse)
async def run_ingestion_qa(
    corpus_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Scan corpus documents for extraction/quality issues without mutating text."""
    run = await IngestionQaService(db).run_corpus_qa(corpus_id, user_id=current_user.id)
    return _run_response(run)


@router.get("/documents/{document_id}/ingestion-qa")
async def get_document_ingestion_qa(
    document_id: str,
    run_id: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Inspect QA findings for one document from a (latest) ingestion QA run."""
    return await IngestionQaService(db).get_document_qa_slice(
        document_id, user_id=current_user.id, run_id=run_id
    )


# ------------------------------------------------------------------
# Document cleaning profiles
# ------------------------------------------------------------------


@router.post(
    "/projects/{project_id}/cleaning-profiles",
    response_model=CleaningProfileResponse,
    status_code=201,
)
async def create_cleaning_profile(
    project_id: str,
    body: CleaningProfileCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    profile = await CleaningProfileService(db).create_profile(
        project_id=project_id,
        user_id=current_user.id,
        name=body.name,
        description=body.description,
        version=body.version,
        config=body.config,
    )
    return _cleaning_profile_response(profile)


@router.get(
    "/projects/{project_id}/cleaning-profiles",
    response_model=list[CleaningProfileResponse],
)
async def list_cleaning_profiles(
    project_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    profiles = await CleaningProfileService(db).list_profiles(
        project_id=project_id, user_id=current_user.id
    )
    return [_cleaning_profile_response(p) for p in profiles]


@router.get("/cleaning-profiles/{profile_id}", response_model=CleaningProfileResponse)
async def get_cleaning_profile(
    profile_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    profile = await CleaningProfileService(db).get_profile(profile_id, user_id=current_user.id)
    return _cleaning_profile_response(profile)


@router.patch("/cleaning-profiles/{profile_id}", response_model=CleaningProfileResponse)
async def update_cleaning_profile(
    profile_id: str,
    body: CleaningProfileUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    profile = await CleaningProfileService(db).update_profile(
        profile_id,
        user_id=current_user.id,
        name=body.name,
        description=body.description,
        version=body.version,
        config=body.config,
    )
    return _cleaning_profile_response(profile)


@router.delete("/cleaning-profiles/{profile_id}", status_code=204)
async def delete_cleaning_profile(
    profile_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    await CleaningProfileService(db).delete_profile(profile_id, user_id=current_user.id)


@router.post("/cleaning/preview", response_model=CleaningPreviewResponse)
async def preview_cleaning_config(
    body: CleaningPreviewRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await CleaningProfileService(db).preview(
        user_id=current_user.id,
        project_id=body.project_id,
        texts=body.texts,
        document_id=body.document_id,
        config=body.config,
        cleaning_profile_id=body.cleaning_profile_id,
    )


@router.post("/corpora/{corpus_id}/clean", response_model=AnalysisRunResponse)
async def apply_document_cleaning(
    corpus_id: str,
    body: ApplyCleaningRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Apply a cleaning profile to corpus documents. Raw extracts stay intact."""
    run = await CleaningProfileService(db).apply_to_corpus(
        corpus_id,
        user_id=current_user.id,
        cleaning_profile_id=body.cleaning_profile_id,
        document_ids=body.document_ids,
    )
    return _run_response(run)


# ------------------------------------------------------------------
# Segmentation
# ------------------------------------------------------------------


@router.post("/corpora/{corpus_id}/segment", response_model=AnalysisRunResponse, status_code=202)
async def segment_corpus(
    corpus_id: str,
    body: SegmentRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    run = await SegmentationService(db).start_segmentation(
        corpus_id, user_id=current_user.id, unit_type=body.unit_type
    )
    return _run_response(run)


# ------------------------------------------------------------------
# Preprocessing profiles
# ------------------------------------------------------------------


@router.post(
    "/projects/{project_id}/preprocessing-profiles",
    response_model=PreprocessingProfileResponse,
    status_code=201,
)
async def create_preprocessing_profile(
    project_id: str,
    body: PreprocessingProfileCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    profile = await PreprocessingProfileService(db).create_profile(
        project_id=project_id,
        user_id=current_user.id,
        name=body.name,
        description=body.description,
        config=body.config,
    )
    return _profile_response(profile)


@router.get(
    "/projects/{project_id}/preprocessing-profiles",
    response_model=list[PreprocessingProfileResponse],
)
async def list_preprocessing_profiles(
    project_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    profiles = await PreprocessingProfileService(db).list_profiles(
        project_id=project_id, user_id=current_user.id
    )
    return [_profile_response(p) for p in profiles]


@router.get("/preprocessing-profiles/{profile_id}", response_model=PreprocessingProfileResponse)
async def get_preprocessing_profile(
    profile_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    profile = await PreprocessingProfileService(db).get_profile(profile_id, user_id=current_user.id)
    return _profile_response(profile)


@router.patch("/preprocessing-profiles/{profile_id}", response_model=PreprocessingProfileResponse)
async def update_preprocessing_profile(
    profile_id: str,
    body: PreprocessingProfileUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    profile = await PreprocessingProfileService(db).update_profile(
        profile_id,
        user_id=current_user.id,
        name=body.name,
        description=body.description,
        config=body.config,
    )
    return _profile_response(profile)


@router.delete("/preprocessing-profiles/{profile_id}", status_code=204)
async def delete_preprocessing_profile(
    profile_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    await PreprocessingProfileService(db).delete_profile(profile_id, user_id=current_user.id)


@router.post("/preprocessing/preview", response_model=PreprocessingPreviewResponse)
async def preview_preprocessing_config(
    body: PreprocessingPreviewRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await PreprocessingProfileService(db).preview(
        user_id=current_user.id,
        project_id=body.project_id,
        corpus_id=body.corpus_id,
        unit_type=body.unit_type,
        texts=body.texts,
        config=body.config,
        preprocessing_profile_id=body.preprocessing_profile_id,
        sample_size=body.sample_size,
    )
