"""Static check that core research routes exist (no app settings import)."""

from __future__ import annotations

import unittest
from pathlib import Path


class ResearchRouteRegistrationTest(unittest.TestCase):
    def test_core_route_paths_present_in_module(self):
        routes_path = Path(__file__).resolve().parents[1] / "api" / "routes.py"
        source = routes_path.read_text(encoding="utf-8")
        for fragment in (
            '/projects/{project_id}/corpora"',
            '/corpora/{corpus_id}/segment"',
            '/corpora/{corpus_id}/ingestion-qa"',
            '/documents/{document_id}/ingestion-qa"',
            '/projects/{project_id}/cleaning-profiles"',
            '/cleaning/preview"',
            '/corpora/{corpus_id}/clean"',
            '/classifiers/train"',
            '/classifiers/dataset-preview"',
            '/corpora/{corpus_id}/dashboard"',
            '/projects/{project_id}/demo-seed"',
            '/classifiers/{model_id}/active-learning/queue"',
            '/classifiers/{model_id}/active-learning/assign"',
            '/robustness/sweep"',
            '/preprocessing/preview"',
            "/contextual-datasets/",
        ):
            self.assertIn(fragment, source)


if __name__ == "__main__":
    unittest.main()
