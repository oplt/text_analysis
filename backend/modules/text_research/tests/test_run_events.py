"""SSE event naming for analysis-run state changes."""

from datetime import UTC, datetime
import unittest

from backend.modules.text_research.api.routes import _run_event_name
from backend.modules.text_research.api.schemas import AnalysisRunResponse


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


class RunEventTests(unittest.TestCase):
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
