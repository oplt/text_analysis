"""Topic-model routes (LATEST-017 split from ``routes.py``).

Included from ``routes.py`` without changing public URLs.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps.auth import get_current_user
from backend.api.deps.db import get_db
from backend.modules.identity_access.models import User
from backend.modules.text_research.api.route_serializers import _run_response
from backend.modules.text_research.api.schemas import (
    AnalysisRunResponse,
    TopicKSweepRequest,
    TopicLabelRequest,
    TopicSeedStabilityRequest,
    TopicTrainRequest,
)
from backend.modules.text_research.application.topic_model_service import TopicModelService

router = APIRouter()


def _topic_filters(body: TopicTrainRequest) -> dict[str, Any]:
    return {
        k: v
        for k, v in body.model_dump().items()
        if k
        not in {
            "unit_type",
            "algorithm",
            "n_topics",
            "preprocessing_profile_id",
            "max_iterations",
            "random_seed",
            "group_by",
            "holdout_fraction",
            "holdout_unit_ids",
            "embedding_provider",
            "embedding_model_name",
            "persist_embedding_artifacts",
            "run_async",
        }
        and v is not None
    }


@router.post(
    "/corpora/{corpus_id}/topics/train", response_model=AnalysisRunResponse, status_code=202
)
async def train_topic_model(
    corpus_id: str,
    body: TopicTrainRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    run = await TopicModelService(db).train(
        corpus_id,
        user_id=current_user.id,
        unit_type=body.unit_type,
        algorithm=body.algorithm,
        n_topics=body.n_topics,
        preprocessing_profile_id=body.preprocessing_profile_id,
        max_iterations=body.max_iterations,
        random_seed=body.random_seed,
        group_by=body.group_by,
        holdout_fraction=body.holdout_fraction,
        holdout_unit_ids=body.holdout_unit_ids,
        embedding_provider=body.embedding_provider,
        embedding_model_name=body.embedding_model_name,
        persist_embedding_artifacts=body.persist_embedding_artifacts,
        run_async=body.run_async,
        **_topic_filters(body),
    )
    return _run_response(run)


def _topic_ksweep_filters(body: TopicKSweepRequest) -> dict[str, Any]:
    return {
        k: v
        for k, v in body.model_dump().items()
        if k
        not in {
            "unit_type",
            "algorithm",
            "k_values",
            "preprocessing_profile_id",
            "max_iterations",
            "random_seed",
            "holdout_fraction",
            "holdout_unit_ids",
            "run_async",
        }
        and v is not None
    }


@router.post(
    "/corpora/{corpus_id}/topics/k-sweep", response_model=AnalysisRunResponse, status_code=202
)
async def topic_k_sweep(
    corpus_id: str,
    body: TopicKSweepRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    run = await TopicModelService(db).run_k_sweep(
        corpus_id,
        user_id=current_user.id,
        unit_type=body.unit_type,
        algorithm=body.algorithm,
        k_values=body.k_values,
        preprocessing_profile_id=body.preprocessing_profile_id,
        max_iterations=body.max_iterations,
        random_seed=body.random_seed,
        holdout_fraction=body.holdout_fraction,
        holdout_unit_ids=body.holdout_unit_ids,
        run_async=body.run_async,
        **_topic_ksweep_filters(body),
    )
    return _run_response(run)


def _topic_seed_stability_filters(body: TopicSeedStabilityRequest) -> dict[str, Any]:
    return {
        k: v
        for k, v in body.model_dump().items()
        if k
        not in {
            "unit_type",
            "algorithm",
            "n_topics",
            "seeds",
            "preprocessing_profile_id",
            "max_iterations",
            "run_async",
        }
        and v is not None
    }


@router.post(
    "/corpora/{corpus_id}/topics/seed-stability",
    response_model=AnalysisRunResponse,
    status_code=202,
)
async def topic_seed_stability(
    corpus_id: str,
    body: TopicSeedStabilityRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    run = await TopicModelService(db).run_seed_stability(
        corpus_id,
        user_id=current_user.id,
        unit_type=body.unit_type,
        algorithm=body.algorithm,
        n_topics=body.n_topics,
        seeds=body.seeds,
        preprocessing_profile_id=body.preprocessing_profile_id,
        max_iterations=body.max_iterations,
        run_async=body.run_async,
        **_topic_seed_stability_filters(body),
    )
    return _run_response(run)


@router.post("/topics/{run_id}/labels", status_code=201)
async def name_topic(
    run_id: str,
    body: TopicLabelRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    label = await TopicModelService(db).name_topic(
        run_id,
        user_id=current_user.id,
        topic_id=body.topic_id,
        human_name=body.human_name,
    )
    return {"id": label.id, "topic_id": label.topic_id, "human_name": label.human_name}


@router.get("/topics/{run_id}/labels")
async def list_topic_labels(
    run_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    labels = await TopicModelService(db).get_topic_labels(run_id, user_id=current_user.id)
    return [
        {
            "id": label.id,
            "topic_id": label.topic_id,
            "human_name": label.human_name,
            "created_at": label.created_at,
        }
        for label in labels
    ]
