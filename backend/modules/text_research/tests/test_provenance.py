"""Tests for strengthened provenance / reproducibility (§23 / Phase 13)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from backend.modules.text_research.application.analysis_executor import attach_run_identity
from backend.modules.text_research.domain.analysis_specification import AnalysisSpecification
from backend.modules.text_research.infrastructure import provenance
from backend.modules.text_research.infrastructure.pipeline_compiler import (
    ENGINE_VERSION,
    compile_plan,
)


class ProvenanceBuilderTests(unittest.TestCase):
    def test_build_run_provenance_includes_required_fields(self) -> None:
        spec = AnalysisSpecification.from_flat(
            corpus_id="corpus-1",
            analysis_type="frequencies",
            unit_type="paragraph",
            random_seed=7,
            preprocessing_profile_id="prep-1",
            cleaning_profile_id="clean-1",
        )
        payload = provenance.build_run_provenance(
            spec=spec,
            corpus_checksum="abc",
            pipeline_checksum="def",
            parent_artifact_checksums=["parent-1"],
            preprocessing_config={"lowercase": True, "language": "en"},
            cleaning_profile={"id": "clean-1", "config": {"strip_html": True}},
            campaign_id="camp-1",
            codebook_id="cb-1",
            codebook_version="3",
        )
        self.assertEqual(payload["schema_version"], "2")
        self.assertEqual(payload["corpus_checksum"], "abc")
        self.assertEqual(payload["pipeline_checksum"], "def")
        self.assertEqual(payload["corpus_snapshot_hash"], "abc")
        self.assertEqual(payload["parent_artifact_checksums"], ["parent-1"])
        self.assertEqual(payload["cleaning_profile"]["id"], "clean-1")
        self.assertEqual(payload["cleaning_profile"]["config_hash"], provenance.stable_content_hash({"strip_html": True}))
        self.assertEqual(payload["random_seeds"]["analysis"], 7)
        self.assertIn("numpy", payload["package_versions"])
        self.assertEqual(payload["engine_version"], ENGINE_VERSION)
        self.assertIsNotNone(payload["analysis_specification"])
        self.assertEqual(payload["analysis_spec_hash"], spec.spec_hash())
        self.assertIn("implementation_version", payload)
        self.assertIn("python_version", payload)
        self.assertEqual(payload["unit_type"], "paragraph")
        self.assertEqual(payload["campaign_id"], "camp-1")
        self.assertEqual(payload["codebook_id"], "cb-1")
        self.assertEqual(payload["codebook_version"], "3")
        self.assertIsNotNone(payload["preprocessing_config_hash"])

    def test_runtime_environment_includes_phase13_identity(self) -> None:
        runtime = provenance.runtime_environment()
        self.assertIn("git_commit", runtime)
        self.assertIn("application_version", runtime)
        self.assertIn("python_version", runtime)
        self.assertRegex(runtime["python_version"], r"^\d+\.\d+\.\d+")
        self.assertIn("package_lock_checksum", runtime)
        self.assertIn("docker_image_digest", runtime)
        self.assertIn("library_versions", runtime)

    def test_package_lock_checksum_hashes_uv_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lock = root / "uv.lock"
            lock.write_text("version = 1\n", encoding="utf-8")
            result = provenance.package_lock_checksum(search_roots=[root])
            assert result is not None
            self.assertEqual(result["path"], "uv.lock")
            self.assertEqual(
                result["sha256"],
                __import__("hashlib").sha256(b"version = 1\n").hexdigest(),
            )

    def test_package_lock_checksum_discovers_committed_backend_lock(self) -> None:
        result = provenance.package_lock_checksum()
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["path"], "uv.lock")
        self.assertEqual(len(result["sha256"]), 64)
        self.assertGreater(result["bytes"], 0)

    def test_partition_hash_is_order_independent(self) -> None:
        self.assertEqual(
            provenance.partition_hash(["b", "a"]),
            provenance.partition_hash(["a", "b"]),
        )
        self.assertIsNone(provenance.partition_hash([]))

    def test_container_image_digest_from_settings_and_env(self) -> None:
        with mock.patch.dict(os.environ, {"RESEARCH_IMAGE_DIGEST": "sha256:deadbeef"}, clear=False):
            self.assertEqual(provenance.container_image_digest(), "sha256:deadbeef")

    def test_attach_run_identity_embeds_provenance(self) -> None:
        spec = AnalysisSpecification.from_flat(
            corpus_id="c1",
            analysis_type="dfm",
            unit_type="document",
            random_seed=3,
        )
        params = attach_run_identity(
            {"corpus_checksum": "corp", "pipeline_checksum": "pipe"},
            spec,
            parent_artifact_checksums=["corp", "pipe"],
        )
        self.assertIn("provenance", params)
        self.assertEqual(params["analysis_spec_hash"], spec.spec_hash())
        self.assertEqual(params["engine_version"], ENGINE_VERSION)
        self.assertEqual(params["provenance"]["corpus_checksum"], "corp")
        self.assertEqual(params["analysis_specification"]["analysis"]["type"], "dfm")
        self.assertEqual(params["provenance"]["schema_version"], "2")

    def test_merge_completion_provenance_preserves_and_extends(self) -> None:
        params = {
            "provenance": {
                "schema_version": "2",
                "git_commit": "abc",
                "output_artifact_checksums": ["old"],
            }
        }
        merged = provenance.merge_completion_provenance(
            params,
            model_artifact_checksum="model-sha",
            output_artifact_checksums=["old", "new"],
            split_hashes={"train": "t", "test": "x"},
        )
        self.assertEqual(merged["provenance"]["git_commit"], "abc")
        self.assertEqual(merged["provenance"]["model_artifact_checksum"], "model-sha")
        self.assertEqual(merged["provenance"]["output_artifact_checksums"], ["old", "new"])
        self.assertEqual(merged["provenance"]["split_hashes"]["train"], "t")

    def test_enrich_provenance_response_adds_lifecycle_and_artifact_fields(self) -> None:
        run = SimpleNamespace(
            created_by="user-1",
            started_at=None,
            completed_at=None,
            corpus_id="corpus-1",
            project_id="proj-1",
            random_seed=9,
        )
        enriched = provenance.enrich_provenance_response(
            run=run,
            parameters={"provenance": {"schema_version": "2"}},
            results={
                "artifact_metadata": {
                    "model": {"sha256": "m1"},
                    "vectorizer": {"sha256": "v1"},
                },
                "train_groups": ["d1", "d2"],
                "test_groups": ["d3"],
            },
        )
        self.assertEqual(enriched["created_by"], "user-1")
        self.assertEqual(enriched["model_artifact_checksum"], "m1")
        self.assertIn("m1", enriched["output_artifact_checksums"])
        self.assertEqual(
            enriched["split_hashes"]["train"],
            provenance.partition_hash(["d1", "d2"]),
        )

    def test_extract_reproduce_request(self) -> None:
        payload = provenance.extract_reproduce_request(
            {
                "provenance": {
                    "analysis_specification": {"analysis": {"type": "kwic"}},
                    "analysis_spec_hash": "hash",
                    "corpus_checksum": "c",
                    "application_version": "0.1.0",
                }
            },
            run_type="kwic",
            run_id="run-9",
        )
        self.assertEqual(payload["source_run_id"], "run-9")
        self.assertEqual(payload["rerun_path"], "/api/v1/research/runs/run-9/rerun")
        self.assertEqual(payload["analysis_spec_hash"], "hash")
        self.assertEqual(payload["application_version"], "0.1.0")


class StageManifestProvenanceTests(unittest.TestCase):
    def test_stage_runner_manifest_includes_provenance(self) -> None:
        from backend.modules.text_research.infrastructure.stage_runner import (
            _stage_build_manifest,
        )

        spec = AnalysisSpecification.from_flat(
            corpus_id="corpus-1",
            analysis_type="frequencies",
            unit_type="paragraph",
            random_seed=1,
            analysis_parameters={"top_n": 5},
        )
        plan = compile_plan(spec)
        context: dict = {
            "spec": spec.normalize(),
            "checksums": {"corpus_checksum": "corp", "pipeline_checksum": "pipe"},
            "stage_timings": {},
        }
        _stage_build_manifest(context, plan)
        self.assertIn("manifest", context)
        self.assertIn("provenance", context["manifest"])
        self.assertEqual(context["manifest"]["provenance"]["schema_version"], "2")


if __name__ == "__main__":
    unittest.main()
