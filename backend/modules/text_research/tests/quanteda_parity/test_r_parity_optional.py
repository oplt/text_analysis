"""Optional quanteda (R) parity checks — skipped unless explicitly enabled."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import unittest
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"
R_SCRIPT = Path(__file__).parent / "r" / "generate_fixtures.R"


def _r_parity_enabled() -> bool:
    return os.environ.get("QUANTEDA_R_PARITY") == "1" and shutil.which("Rscript") is not None


@unittest.skipUnless(_r_parity_enabled(), "Set QUANTEDA_R_PARITY=1 and install Rscript to run")
class QuantedaROptionalParityTests(unittest.TestCase):
    def test_r_generated_fixtures_align_with_python_reference(self) -> None:
        python_ref = json.loads((FIXTURES / "tiny_corpus.json").read_text())
        out_dir = FIXTURES / "r_generated"
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / "quanteda_tiny_corpus.json"

        subprocess.run(
            ["Rscript", str(R_SCRIPT), str(out_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
        if not out_path.exists():
            self.skipTest("quanteda not installed in R; generator exited without output")

        r_payload = json.loads(out_path.read_text())
        self.assertEqual(r_payload["corpus"], python_ref["corpus"])
        self.assertEqual(r_payload["dfm"]["n_features"], python_ref["dfm"]["n_features"])
        self.assertEqual(
            sorted(r_payload["dfm"]["vocabulary_sorted"]),
            python_ref["dfm"]["vocabulary_sorted"],
        )

    def test_known_differences_documented(self) -> None:
        text = (FIXTURES / "known_differences.md").read_text()
        self.assertIn("CI note", text)
        self.assertIn("workflow_dispatch", text)


class QuantedaROptionalSkipTests(unittest.TestCase):
    def test_skips_when_env_unset(self) -> None:
        if os.environ.get("QUANTEDA_R_PARITY") == "1" and shutil.which("Rscript"):
            self.skipTest("R parity enabled in this environment")
        self.assertFalse(_r_parity_enabled())


if __name__ == "__main__":
    unittest.main()
