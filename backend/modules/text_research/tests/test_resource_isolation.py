"""Tests for container/cgroup resource isolation helpers (§18)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from backend.modules.text_research.domain.analysis_specification import (
    AnalysisSpecification,
    ExecutionSpec,
)
from backend.modules.text_research.infrastructure import provenance
from backend.modules.text_research.infrastructure.resource_isolation import (
    effective_resource_limits,
    read_cgroup_limits,
)


class ResourceIsolationTests(unittest.TestCase):
    def test_read_cgroup_v2_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "memory.max").write_text("4294967296\n", encoding="utf-8")
            (root / "cpu.max").write_text("200000 100000\n", encoding="utf-8")
            (root / "pids.max").write_text("256\n", encoding="utf-8")
            limits = read_cgroup_limits(root=root)
        self.assertEqual(limits["cgroup_version"], 2)
        self.assertEqual(limits["memory_bytes"], 4294967296)
        self.assertEqual(limits["memory_mb"], 4096)
        self.assertEqual(limits["cpu_cores"], 2.0)
        self.assertEqual(limits["pids_max"], 256)

    def test_effective_limits_merge_configured_and_mark_advisory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "memory.max").write_text("2147483648\n", encoding="utf-8")
            (root / "cpu.max").write_text("max\n", encoding="utf-8")
            (root / "pids.max").write_text("max\n", encoding="utf-8")
            with (
                mock.patch.dict(
                    "os.environ",
                    {
                        "RESEARCH_R_WORKER_MEMORY_LIMIT": "4g",
                        "RESEARCH_R_WORKER_CPUS": "2.0",
                        "RESEARCH_R_WORKER_PIDS_LIMIT": "256",
                        "CELERY_R_CONCURRENCY": "1",
                    },
                    clear=False,
                ),
                mock.patch(
                    "backend.modules.text_research.infrastructure.resource_isolation._configured_worker_limits",
                    return_value={
                        "celery_concurrency": 1,
                        "timeout_seconds": 600,
                        "configured_memory_limit": "4g",
                        "configured_cpu_limit": "2.0",
                        "configured_cpu_cores": 2.0,
                        "configured_pids_limit": 256,
                    },
                ),
            ):
                payload = effective_resource_limits(cgroup_root=root)

        self.assertEqual(payload["enforcement"], "container_cgroup")
        self.assertEqual(payload["execution_spec_policy"], "advisory")
        self.assertEqual(payload["celery_concurrency"], 1)
        self.assertEqual(payload["memory_mb"], 2048)
        self.assertEqual(payload["configured_memory_limit"], "4g")
        self.assertIn("advisory", payload["note"])

    def test_execution_spec_is_advisory_in_model_docs(self) -> None:
        spec = ExecutionSpec(cpu=4, memory_mb=8192)
        schema = ExecutionSpec.model_json_schema()
        self.assertIn("not enforced", schema["properties"]["cpu"]["description"].lower())
        self.assertIn("not enforced", schema["properties"]["memory_mb"]["description"].lower())
        dumped = AnalysisSpecification.from_flat(
            corpus_id="c1",
            analysis_type="frequencies",
            execution={"cpu": 4, "memory_mb": 8192},
        )
        self.assertEqual(dumped.execution.cpu, 4)
        self.assertEqual(spec.cpu, 4)

    def test_provenance_includes_resource_limits(self) -> None:
        runtime = provenance.runtime_environment()
        self.assertIn("resource_limits", runtime)
        self.assertEqual(runtime["resource_limits"]["execution_spec_policy"], "advisory")
        self.assertEqual(runtime["resource_limits"]["enforcement"], "container_cgroup")

        payload = provenance.build_run_provenance(
            spec=AnalysisSpecification.from_flat(
                corpus_id="c1",
                analysis_type="frequencies",
                execution={"cpu": 2, "memory_mb": 1024},
            )
        )
        self.assertIn("resource_limits", payload)
        self.assertEqual(payload["resource_limits"]["execution_spec_policy"], "advisory")
        # Spec still records requested hints; enforcement lives in resource_limits.
        self.assertEqual(payload["analysis_specification"]["execution"]["cpu"], 2)


if __name__ == "__main__":
    unittest.main()
