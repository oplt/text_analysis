"""Synthesis worker fail-closed behaviour (RAG-P1-021).

Frozen scope missing revision must refuse execution. Worker failures mark the run
FAILED without clearing the queued evidence_revision_hash.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from fastapi import HTTPException

from backend.modules.identity_access.models import User
from backend.modules.text_research.application.corpus_scope_service import (
    CorpusScopeDocumentBinding,
    CorpusScopeSnapshot,
)
from backend.modules.text_research.application.corpus_synthesis_service import (
    CorpusSynthesisService,
)
from backend.modules.text_research.domain.enums import AnalysisRunStatus


def _scope(*, revision_id: str | None, evidence_hash: str = "queued-evidence-hash"):
    return CorpusScopeSnapshot(
        corpus_id="corp-1",
        project_id="proj-1",
        corpus_name="Corpus",
        rag_document_ids=["rag-1"],
        corpus_document_ids=["cd-1"],
        indexed_rag_document_ids=["rag-1"],
        unavailable_rag_document_ids=[],
        scope_hash="scope-hash",
        index_version="index-v1",
        retrieval_version="retrieval-v1",
        evidence_revision_hash=evidence_hash,
        total_documents=1,
        indexed_count=1,
        unavailable_count=0,
        document_bindings=[
            CorpusScopeDocumentBinding(
                corpus_document_id="cd-1",
                rag_document_id="rag-1",
                availability="indexed",
                index_revision_id=revision_id,
                document_revision="doc-rev-1",
            )
        ],
    )


def _apply_run_fields(run, **fields):
    for key, value in fields.items():
        setattr(run, key, value)
    return run


async def _update_run_if_active(run, **fields):
    return _apply_run_fields(run, **fields)


class SynthesisWorkerFailureTests(unittest.IsolatedAsyncioTestCase):
    def test_freeze_scope_missing_revision_fails_closed(self):
        service = CorpusSynthesisService(MagicMock())
        scope = _scope(revision_id=None)

        with self.assertRaises(HTTPException) as ctx:
            service._freeze_scope(scope)

        self.assertEqual(ctx.exception.status_code, 409)
        self.assertIn("historical_revision_unavailable", str(ctx.exception.detail))

    async def test_frozen_revision_ids_missing_revision_fails(self):
        service = CorpusSynthesisService(MagicMock())
        scope = _scope(revision_id=None)

        with self.assertRaises(HTTPException) as ctx:
            await service.scope_service.frozen_revision_ids(scope)

        self.assertEqual(ctx.exception.status_code, 409)
        self.assertIn("historical_revision_unavailable", str(ctx.exception.detail))

    async def test_execute_synthesis_fails_closed_preserves_queued_evidence_hash(self):
        queued_hash = "queued-evidence-hash-abc"
        run = SimpleNamespace(
            id="run-1",
            status=AnalysisRunStatus.QUEUED.value,
            created_by="user-1",
            corpus_id="corp-1",
            project_id="proj-1",
            parameters_json="{}",
            evidence_revision_hash=queued_hash,
            error_message=None,
            progress_stage="queued",
        )

        db = AsyncMock()
        db.commit = AsyncMock()
        db.execute = AsyncMock(
            return_value=SimpleNamespace(
                scalar_one_or_none=lambda: User(
                    id="user-1", email="u@example.com", password_hash="x"
                )
            )
        )

        service = CorpusSynthesisService(db)
        service.repo = MagicMock()
        service.repo.get_run = AsyncMock(return_value=run)
        service.repo.update_run = AsyncMock(side_effect=_apply_run_fields)
        service.repo.update_run_if_active = AsyncMock(side_effect=_update_run_if_active)

        # Bypass JSON reconstruction; force the fail-closed revision gate.
        service._frozen_scope_from_run = lambda params, _run: _scope(revision_id=None)
        service.scope_service.frozen_revision_ids = AsyncMock(
            side_effect=HTTPException(
                status_code=409,
                detail="historical_revision_unavailable: fixed scope has no exact revisions",
            )
        )

        with self.assertRaises(HTTPException):
            await service.execute_synthesis("run-1")

        self.assertEqual(run.status, AnalysisRunStatus.FAILED.value)
        self.assertEqual(run.progress_stage, "failed")
        self.assertEqual(run.evidence_revision_hash, queued_hash)
        self.assertIsNotNone(run.error_message)
        self.assertIn("historical_revision_unavailable", run.error_message)

        # Guarded failure must not wipe the queued evidence hash.
        failed_calls = [
            call
            for call in service.repo.update_run_if_active.await_args_list
            if call.kwargs.get("status") == AnalysisRunStatus.FAILED.value
        ]
        self.assertEqual(len(failed_calls), 1)
        self.assertNotIn("evidence_revision_hash", failed_calls[0].kwargs)

    async def test_execute_synthesis_runtime_failure_preserves_hash(self):
        queued_hash = "preserve-me"
        scope = _scope(revision_id="rev-1", evidence_hash=queued_hash)
        run = SimpleNamespace(
            id="run-2",
            status=AnalysisRunStatus.QUEUED.value,
            created_by="user-1",
            corpus_id="corp-1",
            project_id="proj-1",
            parameters_json="{}",
            evidence_revision_hash=queued_hash,
            error_message=None,
            progress_stage="queued",
        )

        db = AsyncMock()
        db.commit = AsyncMock()
        db.execute = AsyncMock(
            return_value=SimpleNamespace(
                scalar_one_or_none=lambda: User(
                    id="user-1", email="u@example.com", password_hash="x"
                )
            )
        )

        service = CorpusSynthesisService(db)
        service.repo = MagicMock()
        service.repo.get_run = AsyncMock(return_value=run)
        service.repo.update_run = AsyncMock(side_effect=_apply_run_fields)
        service.repo.update_run_if_active = AsyncMock(side_effect=_update_run_if_active)
        service._frozen_scope_from_run = lambda params, _run: scope
        service.scope_service.frozen_revision_ids = AsyncMock(return_value=["rev-1"])
        service._synthesize_sync = AsyncMock(side_effect=RuntimeError("map phase blew up"))

        with self.assertRaises(RuntimeError):
            await service.execute_synthesis("run-2")

        self.assertEqual(run.status, AnalysisRunStatus.FAILED.value)
        self.assertEqual(run.evidence_revision_hash, queued_hash)
        failed_kwargs = [
            call.kwargs
            for call in service.repo.update_run_if_active.await_args_list
            if call.kwargs.get("status") == AnalysisRunStatus.FAILED.value
        ][0]
        self.assertNotIn("evidence_revision_hash", failed_kwargs)


if __name__ == "__main__":
    unittest.main()
