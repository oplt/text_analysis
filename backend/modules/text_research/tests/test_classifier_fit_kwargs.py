"""Regression coverage for classifier training call-site kwargs."""

from __future__ import annotations

import inspect

from backend.modules.text_research.application import classification_service as cs_mod
from backend.modules.text_research.infrastructure.classifiers import fit_text_classifier


def test_classification_service_fit_kwargs_omit_task_type() -> None:
    """fit_kwargs must not include task_type (passed explicitly in both branches)."""
    source = inspect.getsource(cs_mod.ClassificationService.execute_training)
    marker = 'progress_stage="training"'
    idx = source.find(marker)
    assert idx != -1, "expected training progress stage in execute_training"
    window = source[idx : idx + 1800]
    assert "fit_kwargs = {" in window
    start = window.index("fit_kwargs = {")
    end = window.index("\n            }", start) + len("\n            }")
    fit_block = window[start:end]
    assert '"task_type"' not in fit_block and "'task_type'" not in fit_block, fit_block
    assert "task_type=task_type" in window or "\n                    task_type," in window


def test_fit_text_classifier_with_explicit_task_type_and_shared_kwargs() -> None:
    """Mirrors the service call pattern after removing task_type from fit_kwargs."""
    fit_kwargs = {
        "label_names": None,
        "class_weight": None,
        "C": 1.0,
        "random_seed": 0,
        "X_val_texts": None,
        "y_val": None,
        "groups_test": None,
        "tune_thresholds": False,
        "n_bootstrap": 0,
        "calibration_method": "sigmoid",
    }
    result = fit_text_classifier(
        ["a good text", "another sample", "third example", "fourth unit"],
        ["pos", "neg", "pos", "neg"],
        ["held out text", "second held"],
        ["pos", "neg"],
        task_type="binary",
        algorithm="logistic_regression",
        **fit_kwargs,
    )
    assert isinstance(result, dict)
