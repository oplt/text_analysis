"""Tests for reliability bootstrap CIs, diagnostics, and campaign scoping helpers."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from backend.modules.text_research.infrastructure.reliability import (
    attach_ci,
    bootstrap_ci,
    bootstrap_unit_statistic,
    cohens_kappa,
    fleiss_kappa,
    krippendorff_alpha_nominal,
    raw_agreement,
    reliability_diagnostics,
)


class BootstrapCiTests(unittest.TestCase):
    def test_bootstrap_ci_percentile(self) -> None:
        values = [float(i) for i in range(100)]
        ci = bootstrap_ci(values, confidence_level=0.95, bootstrap_samples=100, random_seed=7)
        assert ci is not None
        self.assertLessEqual(ci["lower"], ci["upper"])
        self.assertEqual(ci["random_seed"], 7)
        self.assertAlmostEqual(ci["lower"], 2.0, places=5)
        self.assertAlmostEqual(ci["upper"], 97.0, places=5)

    def test_bootstrap_unit_statistic_reproducible(self) -> None:
        pairs = [("yes", "yes"), ("yes", "no"), ("no", "no"), ("yes", "yes")] * 8

        def stat(sample: list[tuple[str, str]]) -> float:
            return raw_agreement([a for a, _ in sample], [b for _, b in sample])

        left = bootstrap_unit_statistic(pairs, stat, bootstrap_samples=300, random_seed=11)
        right = bootstrap_unit_statistic(pairs, stat, bootstrap_samples=300, random_seed=11)
        self.assertEqual(left, right)
        assert left is not None
        self.assertIn("lower", left)
        self.assertIn("upper", left)

    def test_attach_ci(self) -> None:
        kappa = cohens_kappa(["yes", "no", "yes"], ["yes", "yes", "yes"])
        with_ci = attach_ci(kappa, {"lower": 0.1, "upper": 0.9})
        assert with_ci is not None
        self.assertEqual(with_ci["ci"]["lower"], 0.1)
        self.assertIn("kappa", with_ci)


class DiagnosticsTests(unittest.TestCase):
    def test_warns_on_small_n_and_fleiss_violation(self) -> None:
        warnings = reliability_diagnostics(
            n_coders=3,
            pairable_unit_count=5,
            missingness=0.4,
            categories_observed=2,
            fleiss={"kappa": None, "reason": "modal design has only 2 coder(s)"},
            kappa=None,
            alpha={"alpha": 0.5, "missingness": 0.4},
        )
        joined = " | ".join(warnings)
        self.assertIn("Warning:", joined)
        self.assertIn("n is very small", joined)
        self.assertIn("missingness is high", joined)
        self.assertIn("Fleiss design assumptions are violated", joined)

    def test_cohen_overlap_and_prevalence_warnings(self) -> None:
        warnings = reliability_diagnostics(
            n_coders=2,
            pairable_unit_count=14,
            missingness=0.1,
            categories_observed=2,
            fleiss=None,
            kappa={"kappa": 0.2, "ci": {"lower": 0.0, "upper": 0.9}},
            alpha={"alpha": 0.2, "missingness": 0.1},
            class_prevalence={"yes": 0.88, "no": 0.12},
        )
        joined = " | ".join(warnings)
        self.assertIn("Only 14 overlapping annotations are available for Cohen's kappa", joined)
        self.assertIn("88% of annotations use one category. Kappa may be unstable", joined)
        self.assertIn("confidence interval is wide", joined)

    def test_fleiss_three_coders(self) -> None:
        # 4 units × 3 raters, two categories
        matrix = [
            ["yes", "yes", "yes"],
            ["no", "no", "yes"],
            ["yes", "no", "yes"],
            ["no", "no", "no"],
        ]
        result = fleiss_kappa(matrix)
        self.assertIsNotNone(result["kappa"])
        self.assertEqual(result["n_raters"], 3)
        alpha = krippendorff_alpha_nominal(matrix)
        self.assertIsNotNone(alpha["alpha"])

    def test_single_category_degenerate_and_warning(self) -> None:
        kappa = cohens_kappa(["yes", "yes", "yes"], ["yes", "yes", "yes"])
        self.assertEqual(kappa["kappa"], 1.0)
        warnings = reliability_diagnostics(
            n_coders=2,
            pairable_unit_count=10,
            missingness=0.0,
            categories_observed=1,
            fleiss=None,
            kappa=kappa,
            alpha={"alpha": None},
        )
        joined = " | ".join(warnings)
        self.assertIn("only one category was observed", joined)

    def test_no_overlap_warning(self) -> None:
        warnings = reliability_diagnostics(
            n_coders=2,
            pairable_unit_count=0,
            missingness=1.0,
            categories_observed=0,
            fleiss=None,
            kappa=None,
            alpha={"alpha": None},
        )
        joined = " | ".join(warnings)
        self.assertIn("overlap is insufficient", joined)


class CampaignScopedReliabilityServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_compute_uses_sql_rows_and_stores_bootstrap_provenance(self) -> None:
        from backend.modules.text_research.application.reliability_service import ReliabilityService

        label = SimpleNamespace(id="lab-1", name="Universalism")
        codebook = SimpleNamespace(id="cb-1", version="1.0", project_id="proj-1")
        corpus = SimpleNamespace(id="corp-1", project_id="proj-1")
        campaign = SimpleNamespace(
            id="camp-1",
            corpus_id="corp-1",
            project_id="proj-1",
            unit_type="paragraph",
            codebook_id="cb-1",
            annotator_ids_json='["a1","a2"]',
            name="Wave 1",
            annotation_mode="blind_reliability",
            blind_mode=True,
        )

        created: dict = {}

        async def create_run(run):
            created["run"] = run
            return run

        service = ReliabilityService(MagicMock())
        service.get_corpus_or_404 = AsyncMock(return_value=corpus)
        service.get_codebook_or_404 = AsyncMock(return_value=codebook)
        service.ensure_project_access = AsyncMock()
        service.repo = MagicMock()
        service.repo.get_annotation_campaign = AsyncMock(return_value=campaign)
        service.repo.list_labels = AsyncMock(return_value=[label])
        service.repo.list_reliability_annotation_rows = AsyncMock(
            return_value=[
                ("u1", "lab-1", "a1", "yes"),
                ("u1", "lab-1", "a2", "yes"),
                ("u2", "lab-1", "a1", "no"),
                ("u2", "lab-1", "a2", "yes"),
                ("u3", "lab-1", "a1", "yes"),
                ("u3", "lab-1", "a2", "yes"),
            ]
        )
        service.repo.create_run = AsyncMock(side_effect=create_run)
        service.db = MagicMock()
        service.db.commit = AsyncMock()

        run = await service.compute_reliability(
            "corp-1",
            user_id="user-1",
            codebook_id="cb-1",
            campaign_id="camp-1",
            bootstrap_samples=200,
            confidence_level=0.95,
            random_seed=42,
        )

        kwargs = service.repo.list_reliability_annotation_rows.await_args.kwargs
        self.assertEqual(kwargs["campaign_id"], "camp-1")
        self.assertEqual(kwargs["unit_type"], "paragraph")
        self.assertEqual(kwargs["annotator_ids"], ["a1", "a2"])
        self.assertEqual(kwargs["codebook_version"], "1.0")

        import json

        params = json.loads(created["run"].parameters_json)
        self.assertEqual(params["campaign_id"], "camp-1")
        self.assertEqual(params["bootstrap_samples"], 200)
        self.assertEqual(params["random_seed"], 42)
        self.assertEqual(params["provenance"]["extra"]["blind_mode"], True)

        metrics = json.loads(created["run"].metrics_json)
        self.assertEqual(metrics["bootstrap"]["bootstrap_samples"], 200)
        results = json.loads(created["run"].results_json)
        label_row = results["by_label"]["Universalism"]
        self.assertIn("ci", label_row["cohens_kappa"])
        self.assertIn("ci", label_row["krippendorff_alpha"])
        self.assertEqual(label_row["n_coders"], 2)
        self.assertIsNone(label_row["fleiss_kappa"])
        self.assertEqual(run.random_seed, 42)

    async def test_compute_three_coders_fleiss_and_alpha(self) -> None:
        from backend.modules.text_research.application.reliability_service import ReliabilityService

        label = SimpleNamespace(id="lab-1", name="Universalism")
        codebook = SimpleNamespace(id="cb-1", version="1.0", project_id="proj-1")
        corpus = SimpleNamespace(id="corp-1", project_id="proj-1")
        campaign = SimpleNamespace(
            id="camp-1",
            corpus_id="corp-1",
            project_id="proj-1",
            unit_type="paragraph",
            codebook_id="cb-1",
            annotator_ids_json='["a1","a2","a3"]',
            name="Wave 1",
            annotation_mode="blind_reliability",
            blind_mode=True,
        )
        created: dict = {}

        async def create_run(run):
            created["run"] = run
            return run

        service = ReliabilityService(MagicMock())
        service.get_corpus_or_404 = AsyncMock(return_value=corpus)
        service.get_codebook_or_404 = AsyncMock(return_value=codebook)
        service.ensure_project_access = AsyncMock()
        service.repo = MagicMock()
        service.repo.get_annotation_campaign = AsyncMock(return_value=campaign)
        service.repo.list_labels = AsyncMock(return_value=[label])
        service.repo.list_reliability_annotation_rows = AsyncMock(
            return_value=[
                ("u1", "lab-1", "a1", "yes"),
                ("u1", "lab-1", "a2", "yes"),
                ("u1", "lab-1", "a3", "yes"),
                ("u2", "lab-1", "a1", "no"),
                ("u2", "lab-1", "a2", "no"),
                ("u2", "lab-1", "a3", "yes"),
                ("u3", "lab-1", "a1", "yes"),
                ("u3", "lab-1", "a2", "no"),
                ("u3", "lab-1", "a3", "yes"),
            ]
        )
        service.repo.create_run = AsyncMock(side_effect=create_run)
        service.db = MagicMock()
        service.db.commit = AsyncMock()

        await service.compute_reliability(
            "corp-1",
            user_id="user-1",
            codebook_id="cb-1",
            campaign_id="camp-1",
            bootstrap_samples=50,
            confidence_level=0.95,
            random_seed=7,
        )

        import json

        results = json.loads(created["run"].results_json)
        label_row = results["by_label"]["Universalism"]
        self.assertEqual(label_row["n_coders"], 3)
        self.assertIsNotNone(label_row["fleiss_kappa"])
        self.assertIsNotNone(label_row["fleiss_kappa"]["kappa"])
        self.assertIsNotNone(label_row["krippendorff_alpha"]["alpha"])
        self.assertIsNone(label_row["cohens_kappa"])

    async def test_compute_isolates_campaign_unit_type_and_codebook_version(self) -> None:
        from backend.modules.text_research.application.reliability_service import ReliabilityService

        label = SimpleNamespace(id="lab-1", name="Universalism")
        codebook = SimpleNamespace(id="cb-1", version="2.0", project_id="proj-1")
        corpus = SimpleNamespace(id="corp-1", project_id="proj-1")
        campaign = SimpleNamespace(
            id="camp-target",
            corpus_id="corp-1",
            project_id="proj-1",
            unit_type="sentence",
            codebook_id="cb-1",
            annotator_ids_json='["a1","a2"]',
            name="Target",
            annotation_mode="blind_reliability",
            blind_mode=True,
        )

        service = ReliabilityService(MagicMock())
        service.get_corpus_or_404 = AsyncMock(return_value=corpus)
        service.get_codebook_or_404 = AsyncMock(return_value=codebook)
        service.ensure_project_access = AsyncMock()
        service.repo = MagicMock()
        service.repo.get_annotation_campaign = AsyncMock(return_value=campaign)
        service.repo.list_labels = AsyncMock(return_value=[label])
        service.repo.list_reliability_annotation_rows = AsyncMock(return_value=[])
        service.repo.create_run = AsyncMock(side_effect=lambda run: run)
        service.db = MagicMock()
        service.db.commit = AsyncMock()

        await service.compute_reliability(
            "corp-1",
            user_id="user-1",
            codebook_id="cb-1",
            campaign_id="camp-target",
            unit_type=None,
            bootstrap_samples=10,
            random_seed=1,
        )

        kwargs = service.repo.list_reliability_annotation_rows.await_args.kwargs
        self.assertEqual(kwargs["campaign_id"], "camp-target")
        self.assertEqual(kwargs["unit_type"], "sentence")
        self.assertEqual(kwargs["codebook_version"], "2.0")
        self.assertNotEqual(kwargs["campaign_id"], "camp-other")
        self.assertNotEqual(kwargs["unit_type"], "paragraph")
        self.assertNotEqual(kwargs["codebook_version"], "1.0")


if __name__ == "__main__":
    unittest.main()
