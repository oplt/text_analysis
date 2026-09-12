"""List/get/clone/rerun for `AnalysisRun` records.

`rerun` re-executes the persisted parameters through the *same* application
service that originally produced the run, so results are always freshly
computed from current data rather than copied — this matters because
annotations, documents, or models may have changed since the original run.

Exact reproduction is gated by the versioned adapter registry in
``run_adapters``: unsupported or incomplete runs raise with an explicit reason.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from backend.modules.text_research.application.access import ResearchAccessMixin
from backend.modules.text_research.application.run_adapters import (
    capability_for_run,
    execute_rerun,
)
from backend.modules.text_research.domain.models import AnalysisRun, loads


class RunService(ResearchAccessMixin):
    async def list_runs(
        self,
        *,
        project_id: str,
        user_id: str,
        corpus_id: str | None = None,
        run_type: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[AnalysisRun], int]:
        await self.ensure_project_access(user_id=user_id, project_id=project_id)
        return await self.repo.list_runs(
            project_id, corpus_id=corpus_id, run_type=run_type, limit=limit, offset=offset
        )

    async def get_run(self, run_id: str, *, user_id: str) -> AnalysisRun:
        return await self.get_run_or_404(run_id, user_id=user_id)

    async def clone_parameters(self, run_id: str, *, user_id: str) -> dict[str, Any]:
        from backend.modules.text_research.infrastructure.provenance import (
            extract_reproduce_request,
        )

        run = await self.get_run_or_404(run_id, user_id=user_id)
        parameters = loads(run.parameters_json, {})
        reproduce = extract_reproduce_request(parameters, run_type=run.run_type, run_id=run.id)
        capability = capability_for_run(run, parameters)
        return {
            "run_type": run.run_type,
            "corpus_id": run.corpus_id,
            "parameters": parameters,
            "reproduce": reproduce,
            "analysis_specification": reproduce.get("analysis_specification"),
            "analysis_spec_hash": reproduce.get("analysis_spec_hash"),
            "rerunnable": capability.rerunnable,
            "rerun_block_reason": capability.block_reason,
        }

    async def get_provenance(self, run_id: str, *, user_id: str) -> dict[str, Any]:
        from backend.modules.text_research.infrastructure.provenance import (
            enrich_provenance_response,
            extract_reproduce_request,
            runtime_environment,
        )

        run = await self.get_run_or_404(run_id, user_id=user_id)
        parameters = loads(run.parameters_json, {})
        results = loads(run.results_json, {}) if run.results_json else {}
        provenance = enrich_provenance_response(
            run=run,
            parameters=parameters,
            results=results if isinstance(results, dict) else {},
        )
        capability = capability_for_run(run, parameters)
        return {
            "run_id": run.id,
            "run_type": run.run_type,
            "status": run.status,
            "random_seed": run.random_seed,
            "artifact_path": run.artifact_path,
            "created_by": run.created_by,
            "started_at": run.started_at.isoformat() if run.started_at else None,
            "completed_at": run.completed_at.isoformat() if run.completed_at else None,
            "corpus_id": run.corpus_id,
            "project_id": run.project_id,
            "provenance": provenance,
            "reproduce": extract_reproduce_request(
                parameters, run_type=run.run_type, run_id=run.id
            ),
            "runtime_now": runtime_environment(),
            "rerunnable": capability.rerunnable,
            "rerun_block_reason": capability.block_reason,
        }

    async def rerun(self, run_id: str, *, user_id: str, run_async: bool = False) -> AnalysisRun:
        """One-click reproducible re-execution using the original run parameters."""
        run = await self.get_run_or_404(run_id, user_id=user_id)
        return await execute_rerun(self.db, run, user_id=user_id, run_async=run_async)

    async def cancel_run(self, run_id: str, *, user_id: str) -> AnalysisRun:
        import logging
        from datetime import UTC, datetime

        from backend.modules.text_research.application.run_lifecycle import (
            ACTIVE_RUN_STATUSES,
            cancel_if_active,
        )
        from backend.modules.text_research.domain.enums import AnalysisRunStatus

        logger = logging.getLogger(__name__)

        run = await self.get_run_or_404(run_id, user_id=user_id)
        if run.status not in ACTIVE_RUN_STATUSES:
            raise HTTPException(
                status_code=409,
                detail=f"Cannot cancel run in status '{run.status}'",
            )

        # Soft-revoke only not-started / queued Celery tasks; never hard-kill
        # an already-running worker (terminate=False).
        if run.status == AnalysisRunStatus.QUEUED.value and getattr(run, "celery_task_id", None):
            try:
                from backend.workers.celery_app import celery_app

                celery_app.control.revoke(run.celery_task_id, terminate=False)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Failed to revoke Celery task %s for run %s",
                    run.celery_task_id,
                    run.id,
                    exc_info=True,
                )

        cancelled = await cancel_if_active(
            self.repo,
            run,
            cancellation_requested=True,
            progress_stage="cancelled",
            completed_at=datetime.now(UTC),
            error_message="Cancelled by user",
        )
        if cancelled is None:
            refreshed = await self.repo.get_run(run.id)
            status = refreshed.status if refreshed is not None else run.status
            raise HTTPException(
                status_code=409,
                detail=f"Cannot cancel run in status '{status}'",
            )

        if cancelled.artifact_namespace:
            from backend.modules.text_research.infrastructure.model_storage import ARTIFACT_ROOT

            namespace = ARTIFACT_ROOT / cancelled.artifact_namespace
            if namespace.is_dir():
                import shutil

                shutil.rmtree(namespace)
        await self.db.commit()
        refreshed = await self.repo.get_run(cancelled.id)
        assert refreshed is not None
        return refreshed

    async def compare_runs(self, run_a_id: str, run_b_id: str, *, user_id: str) -> dict[str, Any]:
        run_a = await self.get_run_or_404(run_a_id, user_id=user_id)
        run_b = await self.get_run_or_404(run_b_id, user_id=user_id)
        params_a = loads(run_a.parameters_json, {}) or {}
        params_b = loads(run_b.parameters_json, {}) or {}
        metrics_a = loads(run_a.metrics_json, {}) or {}
        metrics_b = loads(run_b.metrics_json, {}) or {}

        sorted(set(params_a) | set(params_b))
        metric_keys = sorted(set(metrics_a) | set(metrics_b))

        def flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
            if isinstance(value, dict):
                for key, nested in value.items():
                    flatten(f"{prefix}.{key}" if prefix else str(key), nested, out)
            else:
                out[prefix] = value

        flat_a: dict[str, Any] = {}
        flat_b: dict[str, Any] = {}
        flatten("", params_a, flat_a)
        flatten("", params_b, flat_b)
        all_param_keys = sorted(set(flat_a) | set(flat_b))

        parameter_diff = [
            {
                "parameter": key,
                "run_a": flat_a.get(key),
                "run_b": flat_b.get(key),
                "changed": flat_a.get(key) != flat_b.get(key),
            }
            for key in all_param_keys
        ]
        metric_diff = [
            {
                "metric": key,
                "run_a": metrics_a.get(key),
                "run_b": metrics_b.get(key),
                "changed": metrics_a.get(key) != metrics_b.get(key),
            }
            for key in metric_keys
        ]
        return {
            "run_a": {
                "id": run_a.id,
                "run_type": run_a.run_type,
                "status": run_a.status,
                "created_at": run_a.created_at.isoformat() if run_a.created_at else None,
                "artifact_path": run_a.artifact_path,
                "random_seed": run_a.random_seed,
            },
            "run_b": {
                "id": run_b.id,
                "run_type": run_b.run_type,
                "status": run_b.status,
                "created_at": run_b.created_at.isoformat() if run_b.created_at else None,
                "artifact_path": run_b.artifact_path,
                "random_seed": run_b.random_seed,
            },
            "parameter_diff": parameter_diff,
            "metric_diff": metric_diff,
            "changed_parameters": [row for row in parameter_diff if row["changed"]],
            "changed_metrics": [row for row in metric_diff if row["changed"]],
        }
