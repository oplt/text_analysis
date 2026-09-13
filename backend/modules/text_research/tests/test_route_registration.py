"""Static check that core research routes exist (no app settings import)."""

from __future__ import annotations

import unittest
from pathlib import Path


class ResearchRouteRegistrationTest(unittest.TestCase):
    def test_core_route_paths_present_in_module(self):
        # LATEST-017: domain endpoints now live in dedicated router modules
        # included from routes.py, so the source is combined across the
        # aggregator and every split router file it includes.
        api_dir = Path(__file__).resolve().parents[1] / "api"
        router_files = (
            "routes.py",
            "corpus_routes.py",
            "annotation_routes.py",
            "quantitative_routes.py",
            "classification_routes.py",
            "topic_routes.py",
            "run_routes.py",
            "export_routes.py",
            "contextual_routes.py",
        )
        source = "\n".join((api_dir / name).read_text(encoding="utf-8") for name in router_files)
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
