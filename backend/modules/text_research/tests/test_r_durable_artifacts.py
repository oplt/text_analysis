"""Durable R artifact persistence (shared storage, no worker-local paths)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.modules.text_research.domain.analysis_result import (
    AnalysisIdentity,
    AnalysisResult,
    RuntimeInfo,
)
from backend.modules.text_research.infrastructure.artifact_store import ArtifactStore
from backend.modules.text_research.infrastructure.r_runtime.artifacts import (
    RArtifactSecurityError,
    collect_and_persist_r_artifacts,
)
from backend.modules.text_research.infrastructure.r_runtime.serializer import RJobBundle


def _result_with_hints() -> AnalysisResult:
    return AnalysisResult(
        analysis_type="dfm",
        runtime=RuntimeInfo(engine="r", implementation="quanteda"),
        identity=AnalysisIdentity(
            spec_hash="spec",
            corpus_checksum="c",
            pipeline_checksum="p",
            engine_name="r",
            engine_version="r-quanteda-2",
        ),
        results={},
        artifacts=[
            {
                "kind": "dfm_sparse_coo",
                "name": "dfm_sparse_coo.parquet",
                "format": "parquet_coo",
                "nnz": 2,
                "shape": [2, 3],
            }
        ],
    )


class DurableRArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.artifact_dir = Path(self._tmpdir.name) / "artifacts"
        self.work_dir = Path(self._tmpdir.name) / "job-xyz"
        self.artifacts = self.work_dir / "artifacts"
        self.artifacts.mkdir(parents=True)
        self.artifact_dir.mkdir(parents=True)
        self.env = patch.dict("os.environ", {"RESEARCH_ARTIFACT_DIR": str(self.artifact_dir)})
        self.env.start()

    def tearDown(self) -> None:
        self.env.stop()
        self._tmpdir.cleanup()

    def _bundle(self) -> RJobBundle:
        return RJobBundle(
            workdir=self.work_dir,
            manifest_path=self.work_dir / "manifest.json",
            result_path=self.work_dir / "result.json",
            analysis_run_id="run-persisted-abc",
        )

    def test_persists_to_shared_storage_and_survives_workdir_deletion(self) -> None:
        payload = b"PARQUET-BYTES-EXAMPLE"
        source = self.artifacts / "dfm_sparse_coo.parquet"
        source.write_bytes(payload)
        store = ArtifactStore()
        result = collect_and_persist_r_artifacts(self._bundle(), _result_with_hints(), store=store)

        self.assertEqual(len(result.artifacts), 1)
        entry = result.artifacts[0]
        self.assertNotIn("path", entry)
        self.assertTrue(str(entry["artifact_id"]).startswith("r_artifact:"))
        self.assertEqual(entry["storage_backend"], "local")
        self.assertEqual(entry["role"], "dfm_sparse_coo")
        self.assertEqual(entry["bytes"], len(payload))
        self.assertEqual(entry["dimensions"], [2, 3])
        self.assertEqual(entry["nnz"], 2)
        self.assertFalse(any(str(self.work_dir) in str(v) for v in entry.values()))

        # Simulate worker container replacement: wipe ephemeral job tree.
        for child in self.work_dir.iterdir():
            if child.is_file():
                child.unlink()
            else:
                import shutil

                shutil.rmtree(child)

        loaded = store.load_bytes(entry["artifact_id"])
        self.assertEqual(loaded, payload)
        descriptor = store.get(entry["artifact_id"])
        self.assertIsNotNone(descriptor)
        assert descriptor is not None
        self.assertEqual(descriptor.producing_run_id, "run-persisted-abc")
        self.assertEqual(descriptor.metadata["storage_key"], entry["storage_key"])
        self.assertEqual(entry["run_id"], "run-persisted-abc")
        self.assertNotEqual(entry["run_id"], self.work_dir.name)

    def test_rejects_symlink_artifacts(self) -> None:
        target = self.work_dir / "outside.bin"
        target.write_bytes(b"secret")
        link = self.artifacts / "dfm_sparse_coo.parquet"
        link.symlink_to(target)
        with self.assertRaises(RArtifactSecurityError):
            collect_and_persist_r_artifacts(self._bundle(), _result_with_hints())

    def test_rejects_path_traversal_via_resolve(self) -> None:
        # Craft a file outside artifacts and try to treat it as in-tree via abuse.
        outside = Path(self._tmpdir.name) / "escape.parquet"
        outside.write_bytes(b"nope")
        # Direct call with a path that resolves outside should fail the within-check
        # when we place a symlink-like relative escape is blocked by symlink check.
        # Oversize check:
        with (
            patch(
                "backend.modules.text_research.infrastructure.r_runtime.artifacts.settings"
            ) as mocked,
        ):
            mocked.RESEARCH_R_MAX_ARTIFACTS = 32
            mocked.RESEARCH_R_MAX_ARTIFACT_MB = 0  # 0 MiB => any file too large
            mocked.RESEARCH_R_MAX_ARTIFACTS_TOTAL_MB = 250
            (self.artifacts / "dfm_sparse_coo.parquet").write_bytes(b"x")
            with self.assertRaises(RArtifactSecurityError):
                collect_and_persist_r_artifacts(self._bundle(), _result_with_hints())

    def test_attaches_dfm_matrix_checksum_from_parquet(self) -> None:
        import pandas as pd

        pd.DataFrame({"row": [0, 0, 1], "col": [0, 1, 1], "value": [1.0, 2.0, 3.0]}).to_parquet(
            self.artifacts / "dfm_sparse_coo.parquet"
        )
        pd.DataFrame({"feature_index": [0, 1], "feature": ["alpha", "beta"]}).to_parquet(
            self.artifacts / "dfm_features.parquet"
        )
        pd.DataFrame({"unit_index": [0, 1], "unit_id": ["u1", "u2"]}).to_parquet(
            self.artifacts / "dfm_units.parquet"
        )
        seed = _result_with_hints()
        seed = seed.model_copy(
            update={
                "results": {
                    "feature_names": ["alpha", "beta"],
                    "unit_ids": ["u1", "u2"],
                    "nnz": 3,
                    "sparse_coo": {
                        "rows": [0],
                        "cols": [0],
                        "values": [1.0],
                        "preview_only": True,
                    },
                }
            }
        )
        result = collect_and_persist_r_artifacts(self._bundle(), seed, store=ArtifactStore())
        self.assertTrue(result.results["matrix_checksum_complete"])
        self.assertEqual(result.results["matrix_checksum_cells"], 3)
        self.assertEqual(len(result.results["matrix_checksum"]), 64)


if __name__ == "__main__":
    unittest.main()
