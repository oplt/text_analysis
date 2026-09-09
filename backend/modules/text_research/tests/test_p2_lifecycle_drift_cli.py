"""P2 lifecycle, drift monitoring, and CLI tests (no live DB)."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock

from backend.modules.text_research.application.model_lifecycle_service import label_set_hash
from backend.modules.text_research.cli import main
from backend.modules.text_research.domain.enums import ModelLifecycleStatus
from backend.modules.text_research.domain.models import TrainedModel
from backend.modules.text_research.infrastructure.drift_monitoring import (
    build_drift_report,
    feature_presence_drift,
    prediction_distribution_drift,
    score_distribution_drift,
)


class ModelLifecycleEnumTests(unittest.TestCase):
    def test_lifecycle_enum_values(self) -> None:
        self.assertEqual(ModelLifecycleStatus.CANDIDATE.value, "candidate")
        self.assertEqual(ModelLifecycleStatus.PRODUCTION.value, "production")
        self.assertEqual(ModelLifecycleStatus.DEPRECATED.value, "deprecated")

    def test_trained_model_has_lifecycle_fields(self) -> None:
        column_names = {column.key for column in TrainedModel.__table__.columns}
        self.assertIn("lifecycle_status", column_names)
        self.assertIn("lifecycle_notes", column_names)
        self.assertIn("lifecycle_updated_at", column_names)

    def test_label_set_hash_is_order_invariant(self) -> None:
        left = label_set_hash('["b", "a"]')
        right = label_set_hash('["a", "b"]')
        self.assertEqual(left, right)


class DriftMonitoringTests(unittest.TestCase):
    def test_prediction_distribution_drift_identical_is_zero(self) -> None:
        counts = {"yes": 40, "no": 60}
        result = prediction_distribution_drift(counts, counts)
        self.assertAlmostEqual(result["total_variation_distance"], 0.0)
        self.assertAlmostEqual(result["psi_like"], 0.0)

    def test_prediction_distribution_drift_detects_shift(self) -> None:
        baseline = {"yes": 50, "no": 50}
        current = {"yes": 90, "no": 10}
        result = prediction_distribution_drift(baseline, current)
        self.assertGreater(result["total_variation_distance"], 0.3)

    def test_score_distribution_drift_runs(self) -> None:
        baseline = [0.1, 0.2, 0.3, 0.4, 0.5]
        current = [0.6, 0.7, 0.8, 0.9, 1.0]
        result = score_distribution_drift(baseline, current)
        self.assertIn(result["method"], {"ks_2samp", "mean_std_shift"})
        self.assertGreater(result.get("statistic", result.get("mean_shift", 0)), 0)

    def test_feature_presence_jaccard(self) -> None:
        result = feature_presence_drift(["a", "b", "c"], ["b", "c", "d"])
        self.assertAlmostEqual(result["jaccard_similarity"], 0.5)
        self.assertEqual(result["added_terms"], ["d"])
        self.assertEqual(result["removed_terms"], ["a"])

    def test_build_drift_report_combines_sections(self) -> None:
        report = build_drift_report(
            baseline={
                "label_counts": {"yes": 50, "no": 50},
                "scores": [0.1, 0.2],
                "top_terms": ["alpha", "beta"],
            },
            current={
                "label_counts": {"yes": 70, "no": 30},
                "scores": [0.8, 0.9],
                "top_terms": ["beta", "gamma"],
            },
        )
        self.assertEqual(report["summary"]["section_count"], 3)


class CliTests(unittest.TestCase):
    def test_compile_spec_main(self) -> None:
        spec = {
            "corpus": {
                "corpus_id": "corpus-1",
                "unit_type": "paragraph",
                "filters": {"language": "en"},
            },
            "analysis": {"type": "frequencies", "parameters": {}},
        }
        exit_code = main(["compile-spec", "--json", json.dumps(spec)])
        self.assertEqual(exit_code, 0)

    def test_compile_spec_subprocess(self) -> None:
        spec = {
            "corpus": {
                "corpus_id": "corpus-1",
                "unit_type": "paragraph",
                "filters": {"language": "en"},
            },
            "analysis": {"type": "frequencies", "parameters": {}},
        }
        import os
        from pathlib import Path

        project_root = Path(__file__).resolve().parents[4]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(project_root)
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "backend.modules.text_research.cli",
                "compile-spec",
                "--json",
                json.dumps(spec),
            ],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        self.assertEqual(completed.returncode, 0, msg=completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertIn("spec_hash", payload)
        self.assertIn("stages", payload)

    def test_check_drift_cli(self) -> None:
        baseline = {"label_counts": {"yes": 50, "no": 50}}
        current = {"label_counts": {"yes": 60, "no": 40}}
        exit_code = main(
            [
                "check-drift",
                "--baseline",
                self._write_temp_json(baseline),
                "--current",
                self._write_temp_json(current),
            ]
        )
        self.assertEqual(exit_code, 0)

    def _write_temp_json(self, payload: dict) -> str:
        import tempfile
        from pathlib import Path

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(payload, handle)
            name = handle.name
        self.addCleanup(lambda: Path(name).unlink(missing_ok=True))
        return name


class ModelLifecycleServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_set_status_updates_model(self) -> None:
        from backend.modules.text_research.application.model_lifecycle_service import (
            ModelLifecycleService,
        )

        model = TrainedModel(
            project_id="proj-1",
            corpus_id="corpus-1",
            analysis_run_id="run-1",
            training_dataset_snapshot_id="snap-1",
            model_family="logistic_regression",
            task_type="binary",
            label_ids_json='["no","yes"]',
            feature_config_json="{}",
            training_config_json="{}",
            metrics_json="{}",
            model_artifact_path="/tmp/model.joblib",
            vectorizer_artifact_path="/tmp/vec.joblib",
            created_by="user-1",
        )
        model.id = "model-1"

        service = ModelLifecycleService(MagicMock())
        service.get_model_or_404 = AsyncMock(return_value=model)
        service.repo.get_model = AsyncMock(return_value=model)
        service.db.flush = AsyncMock()
        service.db.commit = AsyncMock()

        updated = await service.set_status(
            "model-1",
            user_id="user-1",
            status="production",
            notes="ready for production",
        )
        self.assertEqual(updated.lifecycle_status, "production")
        self.assertEqual(updated.lifecycle_notes, "ready for production")
        self.assertIsNotNone(updated.lifecycle_updated_at)


if __name__ == "__main__":
    unittest.main()
