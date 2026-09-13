"""Robustness/comparative/dashboard + run lifecycle routes
(LATEST-017 split from ``routes.py``).

Included from ``routes.py`` without changing public URLs.
"""

from __future__ import annotations

import asyncio
import contextlib
import json

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps.auth import get_current_user
from backend.api.deps.db import get_db
from backend.core.pagination import (
    PaginatedResponse,
    PaginationParams,
    paginated_response,
    pagination_params,
)
from backend.db.session import SessionLocal
from backend.modules.identity_access.models import User
from backend.modules.text_research.api.route_serializers import _run_event_name, _run_response
from backend.modules.text_research.api.schemas import (
    AnalysisRunResponse,
    ComparativeAnalysisRequest,
    RobustnessRequest,
    RunResultsPageResponse,
)
from backend.modules.text_research.application.comparative_analysis_service import (
    ComparativeAnalysisService,
)
from backend.modules.text_research.application.dashboard_service import DashboardService
from backend.modules.text_research.application.robustness_service import RobustnessService
from backend.modules.text_research.application.run_service import RunService

router = APIRouter()


# ------------------------------------------------------------------
# Robustness, comparative, dashboard
# ------------------------------------------------------------------


@router.post("/robustness/sweep", response_model=AnalysisRunResponse, status_code=202)
async def run_robustness_sweep(
    body: RobustnessRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    run = await RobustnessService(db).run_sweep(
        body.snapshot_id,
        user_id=current_user.id,
        algorithm=body.algorithm,
        seeds=body.seeds,
        cv_folds=body.cv_folds,
        class_weights=body.class_weights,
        test_size=body.test_size,
        group_field=body.group_field,
        max_groups=body.max_groups,
        temporal_field=body.temporal_field,
        temporal_windows=body.temporal_windows,
        transfer_field=body.transfer_field,
        transfer_train_values=body.transfer_train_values,
        transfer_test_values=body.transfer_test_values,
        run_async=body.run_async,
    )
    return _run_response(run)


@router.post("/corpora/{corpus_id}/comparative/prevalence", response_model=AnalysisRunResponse)
async def comparative_prevalence(
    corpus_id: str,
    body: ComparativeAnalysisRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    filters = {
        k: v
        for k, v in body.model_dump().items()
        if k
        not in {
            "unit_type",
            "codebook_id",
            "label_ids",
            "group_by",
            "provenance_mode",
            "model_id",
        }
        and v is not None
    }
    run = await ComparativeAnalysisService(db).prevalence_by_metadata(
        corpus_id,
        user_id=current_user.id,
        unit_type=body.unit_type,
        codebook_id=body.codebook_id,
        label_ids=body.label_ids,
        group_by=body.group_by,
        provenance_mode=body.provenance_mode,
        model_id=body.model_id,
        **filters,
    )
    return _run_response(run)


@router.get("/corpora/{corpus_id}/dashboard")
async def dashboard_summary(
    corpus_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await DashboardService(db).summary(corpus_id, user_id=current_user.id)


# ------------------------------------------------------------------
# Runs
# ------------------------------------------------------------------


@router.get("/projects/{project_id}/runs", response_model=PaginatedResponse[AnalysisRunResponse])
async def list_runs(
    project_id: str,
    corpus_id: str | None = None,
    run_type: str | None = None,
    pagination: PaginationParams = Depends(pagination_params),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    runs, total = await RunService(db).list_runs(
        project_id=project_id,
        user_id=current_user.id,
        corpus_id=corpus_id,
        run_type=run_type,
        limit=pagination.limit,
        offset=pagination.offset,
    )
    return paginated_response(
        [_run_response(r) for r in runs],
        total=total,
        limit=pagination.limit,
        offset=pagination.offset,
    )


@router.get("/runs/{run_id}", response_model=AnalysisRunResponse)
async def get_run(
    run_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    run = await RunService(db).get_run(run_id, user_id=current_user.id)
    return _run_response(run)


@router.get("/runs/{run_id}/results-artifact")
async def get_run_results_artifact(
    run_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return the full result payload when this authorized run was artifactized."""
    from backend.modules.text_research.application.result_artifacts import load_full_run_results

    run = await RunService(db).get_run(run_id, user_id=current_user.id)
    try:
        payload, artifact_id, _descriptor = load_full_run_results(run)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Run result artifact is unavailable.") from exc
    if artifact_id is None:
        raise HTTPException(status_code=404, detail="Run results were not artifactized.")
    return payload


@router.get("/runs/{run_id}/events")
async def stream_run_events(
    run_id: str,
    current_user: User = Depends(get_current_user),
):
    """Stream run state via Redis Pub/Sub, with rare DB reconcile + FE poll fallback.

    Workers persist durable state in PostgreSQL and publish snapshots to Redis.
    This endpoint is the transport only — it does not poll the DB every second.
    """
    from backend.core.cache import redis_client
    from backend.core.config import settings
    from backend.modules.text_research.infrastructure.run_events import (
        TERMINAL_RUN_STATUSES,
        run_events_channel,
    )

    # Authorize once up front.
    async with SessionLocal() as session:
        await RunService(session).get_run(run_id, user_id=current_user.id)

    async def _load_response() -> AnalysisRunResponse:
        async with SessionLocal() as session:
            run = await RunService(session).get_run(run_id, user_id=current_user.id)
            return _run_response(run)

    async def events():
        previous: AnalysisRunResponse | None = None
        try:
            from backend.observability.prometheus_metrics import research_sse_active_connections

            research_sse_active_connections.inc()
        except Exception:
            pass
        try:
            current = await _load_response()
            event_name = _run_event_name(current, previous)
            payload = json.dumps(current.model_dump(mode="json"), separators=(",", ":"))
            if current.artifact_path:
                yield f"event: artifact-created\ndata: {payload}\n\n"
            yield f"event: {event_name}\ndata: {payload}\n\n"
            previous = current
            if current.status in TERMINAL_RUN_STATUSES:
                return

            channel = run_events_channel(run_id)
            pubsub = None
            use_redis = bool(getattr(settings, "CACHE_ENABLED", True))
            last_db_reconcile = asyncio.get_running_loop().time()
            db_reconcile_every = 15.0

            try:
                if use_redis:
                    pubsub = redis_client.pubsub()
                    await pubsub.subscribe(channel)

                while True:
                    message = None
                    if pubsub is not None:
                        try:
                            message = await pubsub.get_message(
                                ignore_subscribe_messages=True, timeout=1.0
                            )
                        except Exception:
                            # Redis hiccup → fall back to DB polling for this stream.
                            pubsub = None
                            use_redis = False

                    if message and message.get("type") == "message":
                        raw = message.get("data")
                        try:
                            envelope = json.loads(raw) if isinstance(raw, str) else raw
                            run_payload = (
                                envelope.get("run") if isinstance(envelope, dict) else None
                            )
                            redis_event = (
                                envelope.get("event") if isinstance(envelope, dict) else None
                            )
                        except (TypeError, json.JSONDecodeError):
                            run_payload = None
                            redis_event = None

                        if isinstance(run_payload, dict):
                            current = AnalysisRunResponse.model_validate(run_payload)
                            payload = json.dumps(
                                current.model_dump(mode="json"), separators=(",", ":")
                            )
                            if current.artifact_path and (
                                previous is None or current.artifact_path != previous.artifact_path
                            ):
                                yield f"event: artifact-created\ndata: {payload}\n\n"
                            name = redis_event or _run_event_name(current, previous)
                            if name != "artifact-created":
                                yield f"event: {name}\ndata: {payload}\n\n"
                            elif (
                                previous is not None
                                and current.artifact_path == previous.artifact_path
                            ):
                                # Redis said artifact-created but path unchanged.
                                yield f"event: progress\ndata: {payload}\n\n"
                            previous = current
                            if current.status in TERMINAL_RUN_STATUSES:
                                return
                            continue

                    now = asyncio.get_running_loop().time()
                    # Rare DB reconcile (missed publish / Redis down) — not 1Hz polling.
                    should_reconcile = (not use_redis) or (
                        now - last_db_reconcile >= db_reconcile_every
                    )
                    if should_reconcile:
                        last_db_reconcile = now
                        current = await _load_response()
                        if previous != current:
                            payload = json.dumps(
                                current.model_dump(mode="json"), separators=(",", ":")
                            )
                            if current.artifact_path and (
                                previous is None or current.artifact_path != previous.artifact_path
                            ):
                                yield f"event: artifact-created\ndata: {payload}\n\n"
                            yield (
                                f"event: {_run_event_name(current, previous)}\ndata: {payload}\n\n"
                            )
                            previous = current
                        if current.status in TERMINAL_RUN_STATUSES:
                            return
                        if not use_redis:
                            await asyncio.sleep(2)
                            continue

                    # SSE comment heartbeat keeps proxies from buffering/closing idle streams.
                    yield ": keepalive\n\n"
            finally:
                if pubsub is not None:
                    with contextlib.suppress(Exception):
                        await pubsub.unsubscribe(channel)
                        await pubsub.aclose()
        finally:
            try:
                from backend.observability.prometheus_metrics import research_sse_active_connections

                research_sse_active_connections.dec()
            except Exception:
                pass

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/runs/{run_id}/clone-parameters")
async def clone_run_parameters(
    run_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await RunService(db).clone_parameters(run_id, user_id=current_user.id)


@router.get("/runs/{run_id}/provenance")
async def get_run_provenance(
    run_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Full provenance block + one-click reproduce payload for a persisted run."""
    return await RunService(db).get_provenance(run_id, user_id=current_user.id)


@router.get("/runs/{run_id}/results", response_model=RunResultsPageResponse)
async def get_run_results(
    run_id: str,
    key: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1_000),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Read a paged result array after authorizing through the owning run."""
    from backend.modules.text_research.application.result_artifacts import page_run_results

    run = await RunService(db).get_run(run_id, user_id=current_user.id)
    try:
        page = page_run_results(run, key=key, limit=limit, offset=offset)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Result artifact is unavailable.") from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Result key was not found.") from exc
    return RunResultsPageResponse(**page.to_dict())


@router.get("/runs/{run_id}/results/download")
async def download_run_results(
    run_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Download the complete managed result payload after run-level authorization."""
    from backend.modules.text_research.application.result_artifacts import load_full_run_results

    run = await RunService(db).get_run(run_id, user_id=current_user.id)
    try:
        payload, artifact_id, descriptor = load_full_run_results(run)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Result artifact is unavailable.") from exc
    checksum = descriptor.checksum if descriptor is not None else None
    headers = {"Content-Disposition": f'attachment; filename="{run_id}-results.json"'}
    if artifact_id:
        headers["X-Results-Artifact-Id"] = artifact_id
    if checksum:
        headers["X-Results-Checksum"] = checksum
    return StreamingResponse(
        iter([json.dumps(payload, default=str)]),
        media_type="application/json",
        headers=headers,
    )


@router.post("/runs/{run_id}/rerun", response_model=AnalysisRunResponse, status_code=202)
async def rerun(
    run_id: str,
    run_async: bool = True,
    exact: bool = False,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """One-click reproducible re-execution of a prior analysis run."""
    run = await RunService(db).rerun(
        run_id, user_id=current_user.id, run_async=run_async, exact=exact
    )
    return _run_response(run)


@router.post("/runs/{run_id}/cancel", response_model=AnalysisRunResponse)
async def cancel_run(
    run_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    run = await RunService(db).cancel_run(run_id, user_id=current_user.id)
    return _run_response(run)


@router.get("/runs/{run_a_id}/compare/{run_b_id}")
async def compare_runs(
    run_a_id: str,
    run_b_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await RunService(db).compare_runs(run_a_id, run_b_id, user_id=current_user.id)
