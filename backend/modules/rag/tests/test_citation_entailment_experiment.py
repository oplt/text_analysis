from __future__ import annotations

import unittest

from backend.modules.rag.eval.citation_entailment_experiment import (
    run_citation_entailment_experiment,
)


class CitationEntailmentExperimentTests(unittest.TestCase):
    def test_disabled_by_default(self):
        report = run_citation_entailment_experiment()
        self.assertTrue(report["skipped"])
        self.assertEqual(report["metrics"], {})

    def test_enabled_run_returns_metrics_dict(self):
        report = run_citation_entailment_experiment(enabled=True)
        self.assertFalse(report["skipped"])
        self.assertIn("mean_overlap", report["metrics"])
        self.assertIn("agreement_rate", report["metrics"])
        self.assertGreaterEqual(report["pair_count"], 2)
