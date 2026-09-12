"""Quantitative, statistical-model, and measurement-validation routes (TASK-022).

Included from ``routes.py`` without changing public URLs.

The request schemas live in ``schemas_quantitative.py``; ``schemas.py``
re-exports them for backwards-compatible imports. Service and frontend splits
remain follow-up work because their orchestration boundaries are still shared.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps.auth import get_current_user
from backend.api.deps.db import get_db
from backend.modules.identity_access.models import User
from backend.modules.text_research.api.schemas import AnalysisRunResponse
from backend.modules.text_research.api.schemas_quantitative import (
    AnalysisRequest,
    ClusteringRequest,
    CooccurrenceRequest,
    DfmRequest,
    DictionaryAnalysisRequest,
    DimensionalityReductionRequest,
    DuplicateDetectionRequest,
    FrequencyRequest,
    KeynessRequest,
    KwicRequest,
    MeasurementComparisonRequest,
    NgramRequest,
    ReadabilityRequest,
    SimilarityRequest,
    StatisticalModelRequest,
)
from backend.modules.text_research.application.measurement_validation_service import (
    MeasurementValidationService,
)
from backend.modules.text_research.application.quantitative_analysis_service import (
    QuantitativeAnalysisService,
)
from backend.modules.text_research.application.statistical_modeling_service import (
    StatisticalModelingService,
)
from backend.modules.text_research.domain.models import AnalysisRun

router = APIRouter()

_run_response: Callable[[AnalysisRun], AnalysisRunResponse] | None = None


def bind_run_response(fn: Callable[[AnalysisRun], AnalysisRunResponse]) -> None:
    """Wire the shared AnalysisRun serializer from the main routes module."""
    global _run_response
    _run_response = fn


def _respond(run: AnalysisRun) -> AnalysisRunResponse:
    if _run_response is None:
        raise RuntimeError("quantitative_routes.bind_run_response was not called")
    return _run_response(run)


def _analysis_filters(body: AnalysisRequest) -> dict[str, Any]:
    excluded = {
        "unit_type",
        "preprocessing_profile_id",
        "top_n",
        "n",
        "weighting",
        "k1",
        "b",
        "smooth_idf",
        "rate_per",
        "skip",
        "group_by",
        "force_sparse_only",
        "trim",
        "run_async",
        "keyword",
        "window_size",
        "case_sensitive",
        "query_mode",
        "query_language",
        "token_attribute",
        "max_matches",
        "dictionary_id",
        "dictionary_terms",
        "hierarchy",
        "exclusions",
        "dictionary_language",
        "association_method",
        "directional",
        "min_frequency",
        "min_count",
        "include_network",
        "method",
        "mode",
        "top_k",
        "min_score",
        "centroid_target",
        "query_text",
        "query_unit_id",
        "embeddings",
        "query_embedding",
        "methods",
        "lexical_threshold",
        "char_ngram_size",
        "use_minhash",
        "minhash_num_perm",
        "minhash_shingle_size",
        "minhash_threshold",
        "max_pairs",
    }
    return {k: v for k, v in body.model_dump().items() if k not in excluded and v is not None}


def _reject_unsupported_async(operation: str, run_async: bool) -> None:
    if run_async:
        raise HTTPException(
            status_code=422,
            detail=f"{operation} does not support asynchronous execution.",
        )


@router.post("/corpora/{corpus_id}/analysis/corpus-stats", response_model=AnalysisRunResponse)
async def corpus_stats(
    corpus_id: str,
    body: AnalysisRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _reject_unsupported_async("corpus_stats", body.run_async)
    run = await QuantitativeAnalysisService(db).corpus_stats(
        corpus_id,
        user_id=current_user.id,
        unit_type=body.unit_type,
        preprocessing_profile_id=body.preprocessing_profile_id,
        **_analysis_filters(body),
    )
    return _respond(run)


@router.post("/corpora/{corpus_id}/analysis/frequencies", response_model=AnalysisRunResponse)
async def frequencies(
    corpus_id: str,
    body: FrequencyRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    run = await QuantitativeAnalysisService(db).frequencies(
        corpus_id,
        user_id=current_user.id,
        unit_type=body.unit_type,
        preprocessing_profile_id=body.preprocessing_profile_id,
        top_n=body.top_n,
        rate_per=body.rate_per,
        group_by=body.group_by,
        run_async=body.run_async,
        **_analysis_filters(body),
    )
    return _respond(run)


@router.post("/corpora/{corpus_id}/analysis/ngrams", response_model=AnalysisRunResponse)
async def ngrams(
    corpus_id: str,
    body: NgramRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    run = await QuantitativeAnalysisService(db).ngrams(
        corpus_id,
        user_id=current_user.id,
        unit_type=body.unit_type,
        n=body.n,
        preprocessing_profile_id=body.preprocessing_profile_id,
        top_n=body.top_n,
        rate_per=body.rate_per,
        skip=body.skip,
        run_async=body.run_async,
        **_analysis_filters(body),
    )
    return _respond(run)


@router.post("/corpora/{corpus_id}/analysis/dfm", response_model=AnalysisRunResponse)
async def dfm(
    corpus_id: str,
    body: DfmRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    run = await QuantitativeAnalysisService(db).dfm(
        corpus_id,
        user_id=current_user.id,
        unit_type=body.unit_type,
        weighting=body.weighting,
        k1=body.k1,
        b=body.b,
        smooth_idf=body.smooth_idf,
        preprocessing_profile_id=body.preprocessing_profile_id,
        force_sparse_only=body.force_sparse_only,
        trim=body.trim.model_dump(exclude_none=True) if body.trim else None,
        run_async=body.run_async,
        **_analysis_filters(body),
    )
    return _respond(run)


@router.post("/corpora/{corpus_id}/analysis/kwic", response_model=AnalysisRunResponse)
async def kwic(
    corpus_id: str,
    body: KwicRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _reject_unsupported_async("kwic", body.run_async)
    run = await QuantitativeAnalysisService(db).kwic(
        corpus_id,
        user_id=current_user.id,
        unit_type=body.unit_type,
        keyword=body.keyword,
        window_size=body.window_size,
        case_sensitive=body.case_sensitive,
        query_mode=body.query_mode,
        query_language=body.query_language,
        token_attribute=body.token_attribute,
        max_matches=body.max_matches,
        preprocessing_profile_id=body.preprocessing_profile_id,
        **_analysis_filters(body),
    )
    return _respond(run)


@router.post("/corpora/{corpus_id}/analysis/dictionary", response_model=AnalysisRunResponse)
async def dictionary_analysis(
    corpus_id: str,
    body: DictionaryAnalysisRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _reject_unsupported_async("dictionary", body.run_async)
    run = await QuantitativeAnalysisService(db).dictionary(
        corpus_id,
        user_id=current_user.id,
        unit_type=body.unit_type,
        dictionary_terms=body.dictionary_terms or [],
        dictionary_id=body.dictionary_id,
        hierarchy=body.hierarchy,
        exclusions=body.exclusions,
        dictionary_language=body.dictionary_language,
        case_sensitive=body.case_sensitive,
        rate_per=body.rate_per,
        group_by=body.group_by,
        preprocessing_profile_id=body.preprocessing_profile_id,
        run_async=body.run_async,
        **_analysis_filters(body),
    )
    return _respond(run)


@router.post("/corpora/{corpus_id}/analysis/keyness", response_model=AnalysisRunResponse)
async def keyness(
    corpus_id: str,
    body: KeynessRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    run = await QuantitativeAnalysisService(db).keyness(
        corpus_id,
        user_id=current_user.id,
        unit_type=body.unit_type,
        filters_a=body.filters_a,
        filters_b=body.filters_b,
        group_field=body.group_field,
        method=body.method,
        correction=body.correction,
        min_frequency=body.min_frequency,
        preprocessing_profile_id=body.preprocessing_profile_id,
        top_n=body.top_n,
        run_async=body.run_async,
    )
    return _respond(run)


@router.post("/corpora/{corpus_id}/analysis/cooccurrence", response_model=AnalysisRunResponse)
async def cooccurrence(
    corpus_id: str,
    body: CooccurrenceRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    run = await QuantitativeAnalysisService(db).cooccurrence(
        corpus_id,
        user_id=current_user.id,
        unit_type=body.unit_type,
        window_size=body.window_size,
        top_n=body.top_n,
        association_method=body.association_method,
        directional=body.directional,
        min_frequency=body.min_frequency,
        min_count=body.min_count,
        include_network=body.include_network,
        preprocessing_profile_id=body.preprocessing_profile_id,
        run_async=body.run_async,
        **_analysis_filters(body),
    )
    return _respond(run)


@router.post("/corpora/{corpus_id}/analysis/similarity", response_model=AnalysisRunResponse)
async def similarity(
    corpus_id: str,
    body: SimilarityRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Document-to-document / unit-to-unit / query-to-document / group-centroid
    similarity (cosine-on-TFIDF, Jaccard, or caller-supplied embeddings)."""
    run = await QuantitativeAnalysisService(db).similarity(
        corpus_id,
        user_id=current_user.id,
        unit_type=body.unit_type,
        method=body.method,
        mode=body.mode,
        top_k=body.top_k,
        min_score=body.min_score,
        group_by=body.group_by,
        centroid_target=body.centroid_target,
        query_text=body.query_text,
        query_unit_id=body.query_unit_id,
        embeddings=body.embeddings,
        query_embedding=body.query_embedding,
        embedding_artifact_id=body.embedding_artifact_id,
        preprocessing_profile_id=body.preprocessing_profile_id,
        run_async=body.run_async,
        **_analysis_filters(body),
    )
    return _respond(run)


@router.post(
    "/corpora/{corpus_id}/analysis/duplicate-detection", response_model=AnalysisRunResponse
)
async def duplicate_detection(
    corpus_id: str,
    body: DuplicateDetectionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Exact / normalized checksum, lexical near-dup, and optional MinHash
    duplicate detection — the same engine ingestion QA uses, run explicitly."""
    run = await QuantitativeAnalysisService(db).duplicate_detection(
        corpus_id,
        user_id=current_user.id,
        unit_type=body.unit_type,
        methods=body.methods,
        lexical_threshold=body.lexical_threshold,
        char_ngram_size=body.char_ngram_size,
        use_minhash=body.use_minhash,
        minhash_num_perm=body.minhash_num_perm,
        minhash_shingle_size=body.minhash_shingle_size,
        minhash_threshold=body.minhash_threshold,
        max_pairs=body.max_pairs,
        run_async=body.run_async,
        **_analysis_filters(body),
    )
    return _respond(run)


@router.post("/corpora/{corpus_id}/analysis/clustering", response_model=AnalysisRunResponse)
async def clustering(
    corpus_id: str,
    body: ClusteringRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    run = await QuantitativeAnalysisService(db).clustering(
        corpus_id,
        user_id=current_user.id,
        unit_type=body.unit_type,
        n_clusters=body.n_clusters,
        algorithm=body.algorithm,
        use_svd=body.use_svd,
        n_svd_components=body.n_svd_components,
        top_terms=body.top_terms,
        random_seed=body.random_seed,
        preprocessing_profile_id=body.preprocessing_profile_id,
        run_async=body.run_async,
        **_analysis_filters(body),
    )
    return _respond(run)


@router.post(
    "/corpora/{corpus_id}/analysis/dimensionality-reduction",
    response_model=AnalysisRunResponse,
)
async def dimensionality_reduction(
    corpus_id: str,
    body: DimensionalityReductionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    run = await QuantitativeAnalysisService(db).dimensionality_reduction(
        corpus_id,
        user_id=current_user.id,
        unit_type=body.unit_type,
        method=body.method,
        n_components=body.n_components,
        random_seed=body.random_seed,
        preprocessing_profile_id=body.preprocessing_profile_id,
        run_async=body.run_async,
        **_analysis_filters(body),
    )
    return _respond(run)


@router.post("/corpora/{corpus_id}/analysis/readability", response_model=AnalysisRunResponse)
async def readability(
    corpus_id: str,
    body: ReadabilityRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _reject_unsupported_async("readability", body.run_async)
    run = await QuantitativeAnalysisService(db).readability(
        corpus_id,
        user_id=current_user.id,
        unit_type=body.unit_type,
        **_analysis_filters(body),
    )
    return _respond(run)


@router.post("/corpora/{corpus_id}/analysis/statistical-model", response_model=AnalysisRunResponse)
async def statistical_model(
    corpus_id: str,
    body: StatisticalModelRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    run = await StatisticalModelingService(db).fit(
        corpus_id,
        user_id=current_user.id,
        model=body.model,
        dependent_var=body.dependent_var,
        independent_vars=body.independent_vars,
        rows=body.rows,
        add_intercept=body.add_intercept,
    )
    return _respond(run)


@router.post(
    "/corpora/{corpus_id}/analysis/measurement-comparison",
    response_model=AnalysisRunResponse,
)
async def measurement_comparison(
    corpus_id: str,
    body: MeasurementComparisonRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    run = await MeasurementValidationService(db).compare(
        corpus_id,
        user_id=current_user.id,
        source_a=body.source_a,
        values_a=body.values_a,
        source_b=body.source_b,
        values_b=body.values_b,
        ids=body.ids,
        value_kind=body.value_kind,
        subgroup=body.subgroup,
    )
    return _respond(run)
