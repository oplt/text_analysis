"""Phase 4: supervised feature selection + leakage guards."""

from __future__ import annotations

import unittest
from io import BytesIO
from unittest.mock import patch

import joblib
from sklearn.pipeline import Pipeline

from backend.modules.text_research.infrastructure import classifiers


class FeatureSelectionConfigTests(unittest.TestCase):
    def test_defaults_disabled(self) -> None:
        cfg = classifiers.FeatureSelectionConfig()
        self.assertEqual(cfg.method, "none")
        self.assertFalse(cfg.enabled)

    def test_rejects_bad_method(self) -> None:
        with self.assertRaises(ValueError):
            classifiers.FeatureSelectionConfig(method="anova")

    def test_from_dict_filters_unknown(self) -> None:
        cfg = classifiers.FeatureSelectionConfig.from_dict(
            {"method": "chi2", "k": 10, "bogus": True}
        )
        self.assertEqual(cfg.method, "chi2")
        self.assertEqual(cfg.k, 10)


class SupervisedSelectionFitTests(unittest.TestCase):
    def _toy(self) -> tuple[list[str], list[str], list[str], list[str]]:
        # Distinct polarity words so chi2/MI have signal.
        train_x = [
            "alpha alpha good",
            "alpha good nice",
            "beta beta bad",
            "beta bad ugly",
            "alpha nice",
            "beta ugly",
        ]
        train_y = ["pos", "pos", "neg", "neg", "pos", "neg"]
        test_x = ["alpha good", "beta bad", "LEAKTOKENONLYINTEST appears"]
        test_y = ["pos", "neg", "pos"]
        return train_x, train_y, test_x, test_y

    def test_chi2_reduces_feature_space(self) -> None:
        train_x, train_y, test_x, test_y = self._toy()
        result = classifiers.fit_text_classifier(
            train_x,
            train_y,
            test_x,
            test_y,
            task_type="binary",
            algorithm="logistic_regression",
            feature_config=classifiers.FeatureConfig(vectorizer="count", max_features=None),
            selection_config=classifiers.FeatureSelectionConfig(method="chi2", k=3),
            tune_thresholds=False,
        )
        space = result["feature_space"]
        self.assertLessEqual(space["after_supervised_selection"], space["after_df_pruning"])
        self.assertEqual(space["after_supervised_selection"], 3)
        self.assertEqual(result["vocabulary_size"], 3)
        self.assertIsInstance(result["vectorizer"], Pipeline)

    def test_mutual_info_runs(self) -> None:
        train_x, train_y, test_x, test_y = self._toy()
        result = classifiers.fit_text_classifier(
            train_x,
            train_y,
            test_x,
            test_y,
            task_type="binary",
            algorithm="multinomial_nb",
            feature_config=classifiers.FeatureConfig(vectorizer="count"),
            selection_config=classifiers.FeatureSelectionConfig(method="mutual_info", k=4),
            tune_thresholds=False,
        )
        self.assertEqual(result["vocabulary_size"], 4)

    def test_l1_runs_and_records_selected_feature_metadata(self) -> None:
        train_x, train_y, test_x, test_y = self._toy()
        result = classifiers.fit_text_classifier(
            train_x,
            train_y,
            test_x,
            test_y,
            task_type="binary",
            feature_config=classifiers.FeatureConfig(vectorizer="count"),
            selection_config=classifiers.FeatureSelectionConfig(method="l1"),
            tune_thresholds=False,
        )
        feature_space = result["feature_space"]
        self.assertEqual(len(feature_space["selected_feature_names"]), result["vocabulary_size"])
        self.assertEqual(len(feature_space["selected_feature_hash"]), 64)

    def test_k_larger_than_vocabulary_is_clamped(self) -> None:
        train_x, train_y, test_x, test_y = self._toy()
        result = classifiers.fit_text_classifier(
            train_x, train_y, test_x, test_y,
            task_type="binary",
            feature_config=classifiers.FeatureConfig(vectorizer="count"),
            selection_config=classifiers.FeatureSelectionConfig(method="chi2", k=10_000),
            tune_thresholds=False,
        )
        self.assertEqual(
            result["feature_space"]["after_supervised_selection"],
            result["feature_space"]["after_df_pruning"],
        )

    def test_percentile_selection(self) -> None:
        train_x, train_y, test_x, test_y = self._toy()
        result = classifiers.fit_text_classifier(
            train_x, train_y, test_x, test_y,
            task_type="binary",
            feature_config=classifiers.FeatureConfig(vectorizer="count"),
            selection_config=classifiers.FeatureSelectionConfig(method="chi2", percentile=50),
            tune_thresholds=False,
        )
        self.assertLess(result["vocabulary_size"], result["feature_space"]["after_df_pruning"])

    def test_none_keeps_plain_vectorizer(self) -> None:
        train_x, train_y, test_x, test_y = self._toy()
        result = classifiers.fit_text_classifier(
            train_x,
            train_y,
            test_x,
            test_y,
            task_type="binary",
            feature_config=classifiers.FeatureConfig(vectorizer="tfidf"),
            selection_config=classifiers.FeatureSelectionConfig(method="none"),
            tune_thresholds=False,
        )
        self.assertNotIsInstance(result["vectorizer"], Pipeline)

    def test_test_only_token_absent_from_vocabulary(self) -> None:
        train_x, train_y, test_x, test_y = self._toy()
        result = classifiers.fit_text_classifier(
            train_x,
            train_y,
            test_x,
            test_y,
            task_type="binary",
            feature_config=classifiers.FeatureConfig(vectorizer="count"),
            selection_config=classifiers.FeatureSelectionConfig(method="chi2", k="all"),
            tune_thresholds=False,
        )
        names = {str(n).lower() for n in result["vectorizer"].get_feature_names_out()}
        self.assertFalse(any("leaktokenonlyintest" in n for n in names))

    def test_test_labels_cannot_change_selected_features(self) -> None:
        train_x, train_y, test_x, test_y = self._toy()
        kwargs = {
            "task_type": "binary",
            "feature_config": classifiers.FeatureConfig(vectorizer="count"),
            "selection_config": classifiers.FeatureSelectionConfig(method="chi2", k=3),
            "tune_thresholds": False,
        }
        original = classifiers.fit_text_classifier(train_x, train_y, test_x, test_y, **kwargs)
        changed = classifiers.fit_text_classifier(
            train_x, train_y, test_x, list(reversed(test_y)), **kwargs
        )
        self.assertEqual(
            original["feature_space"]["selected_feature_names"],
            changed["feature_space"]["selected_feature_names"],
        )

    def test_fitted_selector_round_trips_without_changing_features(self) -> None:
        train_x, train_y, test_x, test_y = self._toy()
        result = classifiers.fit_text_classifier(
            train_x, train_y, test_x, test_y,
            task_type="binary",
            feature_config=classifiers.FeatureConfig(vectorizer="count"),
            selection_config=classifiers.FeatureSelectionConfig(method="mutual_info", k=3),
            tune_thresholds=False,
        )
        serialized = BytesIO()
        joblib.dump(result["vectorizer"], serialized)
        serialized.seek(0)
        reloaded = joblib.load(serialized)
        self.assertEqual(
            list(result["vectorizer"].get_feature_names_out()),
            list(reloaded.get_feature_names_out()),
        )

    def test_selector_fit_never_sees_test_rows(self) -> None:
        train_x, train_y, test_x, test_y = self._toy()
        fit_n_samples: list[int] = []
        real_build = classifiers.build_feature_selector

        def _spy(selection_config, *, task_type="binary", random_seed=42):
            selector = real_build(selection_config, task_type=task_type, random_seed=random_seed)
            if selector is None:
                return None
            original_fit = selector.fit

            def fit(X, y=None):
                fit_n_samples.append(X.shape[0])
                return original_fit(X, y)

            selector.fit = fit  # type: ignore[method-assign]
            return selector

        with patch.object(classifiers, "build_feature_selector", side_effect=_spy):
            classifiers.fit_text_classifier(
                train_x,
                train_y,
                test_x,
                test_y,
                task_type="binary",
                feature_config=classifiers.FeatureConfig(vectorizer="count"),
                selection_config=classifiers.FeatureSelectionConfig(method="chi2", k=5),
                tune_thresholds=False,
            )

        self.assertEqual(fit_n_samples, [len(train_x)])
        # Phase 15: leakage is prevented, never soft-warned as if it occurred.
        from backend.modules.text_research.infrastructure.scientific_warnings import (
            assert_no_leakage_soft_warning,
        )

        assert_no_leakage_soft_warning(
            [
                "Warning: Classifier vocabulary: raw=10, post-min_df=8, selected=5 "
                "(selection=SelectKBest)."
            ]
        )

    def test_nested_cv_refits_selector_per_outer_fold(self) -> None:
        texts = [
            "alpha good",
            "alpha nice",
            "beta bad",
            "beta ugly",
            "alpha fine",
            "beta worse",
            "alpha ok",
            "beta no",
        ]
        y = ["pos", "pos", "neg", "neg", "pos", "neg", "pos", "neg"]
        groups = ["g1", "g1", "g2", "g2", "g3", "g3", "g4", "g4"]
        fit_counts: list[int] = []
        real_build = classifiers.build_feature_selector

        def _spy(selection_config, *, task_type="binary", random_seed=42):
            selector = real_build(selection_config, task_type=task_type, random_seed=random_seed)
            if selector is None:
                return None
            original_fit = selector.fit

            def fit(X, y=None):
                fit_counts.append(X.shape[0])
                return original_fit(X, y)

            selector.fit = fit  # type: ignore[method-assign]
            return selector

        with patch.object(classifiers, "build_feature_selector", side_effect=_spy):
            out = classifiers.nested_grouped_cv_evaluation(
                texts,
                y,
                groups,
                task_type="binary",
                algorithm="logistic_regression",
                feature_config=classifiers.FeatureConfig(vectorizer="count"),
                preprocessing_config=None,
                label_names=None,
                class_weight=None,
                C=1.0,
                random_seed=0,
                outer_splits=2,
                inner_splits=2,
                selection_config=classifiers.FeatureSelectionConfig(method="chi2", k=3),
            )

        self.assertGreaterEqual(len(fit_counts), out["outer_splits"])
        # Each outer fold must fit on fewer rows than the full corpus.
        self.assertTrue(all(n < len(texts) for n in fit_counts))


if __name__ == "__main__":
    unittest.main()
