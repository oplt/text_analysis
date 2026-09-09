"""Tests for strengthened provenance / reproducibility (§23)."""

from __future__ import annotations

import os
import unittest
from unittest import mock

from backend.modules.text_research.application.analysis_executor import attach_run_identity
from backend.modules.text_research.domain.analysis_specification import AnalysisSpecification
from backend.modules.text_research.infrastructure import provenance
from backend.modules.text_research.infrastructure.pipeline_compiler import ENGINE_VERSION
from backend.modules.text_research.infrastructure.stage_runner import StageRunner
from backend.modules.text_research.infrastructure.pipeline_compiler import compile_plan


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
        )
        self.assertEqual(payload["schema_version"], "1")
        self.assertEqual(payload["corpus_checksum"], "abc")
        self.assertEqual(payload["pipeline_checksum"], "def")
        self.assertEqual(payload["parent_artifact_checksums"], ["parent-1"])
        self.assertEqual(payload["cleaning_profile"]["id"], "clean-1")
        self.assertEqual(payload["random_seeds"]["analysis"], 7)
        self.assertIn("numpy", payload["package_versions"])
        self.assertEqual(payload["engine_version"], ENGINE_VERSION)
        self.assertIsNotNone(payload["analysis_specification"])
        self.assertEqual(payload["analysis_spec_hash"], spec.spec_hash())
        self.assertIn("implementation_version", payload)

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

    def test_extract_reproduce_request(self) -> None:
        payload = provenance.extract_reproduce_request(
            {
                "provenance": {
                    "analysis_specification": {"analysis": {"type": "kwic"}},
                    "analysis_spec_hash": "hash",
                    "corpus_checksum": "c",
                }
            },
            run_type="kwic",
            run_id="run-9",
        )
        self.assertEqual(payload["source_run_id"], "run-9")
        self.assertEqual(payload["rerun_path"], "/api/v1/research/runs/run-9/rerun")
        self.assertEqual(payload["analysis_spec_hash"], "hash")


class StageManifestProvenanceTests(unittest.TestCase):
    def test_stage_runner_manifest_includes_provenance(self) -> None:
        spec = AnalysisSpecification.from_flat(
            corpus_id="corpus-1",
            analysis_type="frequencies",
            unit_type="paragraph",
            random_seed=1,
            analysis_parameters={"top_n": 5},
        )
        plan = compile_plan(spec)
        context = {
            "spec": spec.normalize(),
            "texts": ["education policy reform", "trade policy debate"],
            "unit_ids": ["u1", "u2"],
            "config": {"lowercase": True, "remove_stopwords": False},
        }
        StageRunner(plan, context).run()
        manifest = context.get("manifest")
        assert manifest is not None
        self.assertIn("provenance", manifest)
        self.assertEqual(manifest["provenance"]["analysis_spec_hash"], plan.spec_hash)
        self.assertTrue(manifest["provenance"]["corpus_checksum"])
        self.assertTrue(manifest["provenance"]["pipeline_checksum"])
        self.assertIn("package_versions", manifest["provenance"])
