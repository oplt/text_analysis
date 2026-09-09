"""Redis-driven analysis-run event transport tests."""

from __future__ import annotations

import json
import unittest
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from backend.modules.text_research.api.routes import _run_event_name
from backend.modules.text_research.api.schemas import AnalysisRunResponse
from backend.modules.text_research.domain.models import AnalysisRun
from backend.modules.text_research.infrastructure import run_events


def _run(*, status: str, stage: str | None = None, artifact_path: str | None = None):
    return AnalysisRunResponse(
        id="run-1",
        project_id="project-1",
        corpus_id="corpus-1",
        run_type="topic_model",
        status=status,
        progress_stage=stage,
        artifact_path=artifact_path,
        random_seed=None,
        created_by="user-1",
        started_at=None,
        completed_at=None,
        error_message=None,
        created_at=datetime.now(UTC),
    )


def _orm_run(**overrides) -> AnalysisRun:
    run = AnalysisRun(
        id="run-1",
        project_id="project-1",
        corpus_id="corpus-1",
        run_type="topic_model",
        status="running",
        progress_stage="training",
        created_by="user-1",
    )
    for key, value in overrides.items():
        setattr(run, key, value)
    return run


class RunEventNamingTests(unittest.TestCase):
    def test_initial_and_terminal_states_use_named_events(self):
        self.assertEqual(_run_event_name(_run(status="queued"), None), "queued")
        self.assertEqual(_run_event_name(_run(status="running"), None), "started")
        self.assertEqual(_run_event_name(_run(status="completed"), None), "completed")
        self.assertEqual(_run_event_name(_run(status="failed"), None), "failed")
        self.assertEqual(_run_event_name(_run(status="cancelled"), None), "cancelled")

    def test_artifact_and_stage_changes_are_reported(self):
        previous = _run(status="running", stage="training")
        self.assertEqual(
            _run_event_name(
                _run(status="running", stage="saving", artifact_path="runs/run-1/model.joblib"),
                previous,
            ),
            "artifact-created",
        )
        self.assertEqual(
            _run_event_name(_run(status="running", stage="saving"), previous), "progress"
        )


class RunEventPublishTests(unittest.TestCase):
    def setUp(self) -> None:
        run_events.reset_run_events_redis_for_tests()

    def tearDown(self) -> None:
        run_events.reset_run_events_redis_for_tests()

    def test_channel_name(self):
        self.assertEqual(
            run_events.run_events_channel("abc"),
            "research:run:abc:events",
        )

    def test_serialize_run_matches_response_shape(self):
        payload = run_events.serialize_run(_orm_run(status="queued", progress_stage="queued"))
        self.assertEqual(payload["id"], "run-1")
        self.assertEqual(payload["status"], "queued")
        self.assertIn("parameters", payload)
        self.assertIn("created_at", payload)

    def test_publish_uses_redis_channel(self):
        client = MagicMock()
        run = _orm_run(status="running", progress_stage="saving")
        with patch.object(run_events, "_get_sync_redis", return_value=client):
            name = run_events.publish_run_event(
                run,
                previous={"status": "running", "progress_stage": "training", "artifact_path": None},
            )
        self.assertEqual(name, "progress")
        client.publish.assert_called_once()
        channel, raw = client.publish.call_args.args
        self.assertEqual(channel, "research:run:run-1:events")
        envelope = json.loads(raw)
        self.assertEqual(envelope["event"], "progress")
        self.assertEqual(envelope["run"]["progress_stage"], "saving")

    def test_publish_without_redis_still_returns_event_name(self):
        with patch.object(run_events, "_get_sync_redis", return_value=None):
            name = run_events.publish_run_event(_orm_run(status="completed"), previous=None)
        self.assertEqual(name, "completed")


class RunEventStreamFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_falls_back_to_db_when_redis_disabled(self) -> None:
        """CACHE_ENABLED=False → SSE still emits DB-reconciled terminal event."""
        from types import SimpleNamespace
        from unittest.mock import patch

        from backend.modules.text_research.api import routes

        queued = _run(status="queued")
        completed = _run(status="completed")
        loads = [queued, completed]

        class FakeSession:
            async def __aenter__(self):
                return MagicMock()

            async def __aexit__(self, *args):
                return False

        class FakeRunService:
            def __init__(self, _db):
                pass

            async def get_run(self, run_id, *, user_id):
                return SimpleNamespace(id=run_id)

        user = SimpleNamespace(id="user-1")

        with (
            patch.object(routes, "SessionLocal", FakeSession),
            patch.object(routes, "RunService", FakeRunService),
            patch("backend.core.config.settings.CACHE_ENABLED", False),
            patch.object(routes, "_run_response", side_effect=lambda _run_obj: loads.pop(0)),
        ):
            response = await routes.stream_run_events(run_id="run-1", current_user=user)
            chunks: list[str] = []
            async for chunk in response.body_iterator:
                text = chunk.decode() if isinstance(chunk, bytes) else str(chunk)
                chunks.append(text)
                if "event: completed" in text:
                    break
                if len(chunks) > 20:
                    break

        joined = "".join(chunks)
        self.assertIn("event: queued", joined)
        self.assertIn("event: completed", joined)


if __name__ == "__main__":
    unittest.main()
