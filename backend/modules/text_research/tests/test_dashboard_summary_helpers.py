"""Unit coverage for dashboard summary enrichment helpers."""

from __future__ import annotations

from backend.modules.text_research.application.dashboard_service import _pick_macro_f1


def test_pick_macro_f1_from_flat_metrics() -> None:
    assert _pick_macro_f1({"macro_f1": 0.82}) == 0.82
    assert _pick_macro_f1({"f1_macro": 0.5}) == 0.5


def test_pick_macro_f1_from_classification_report() -> None:
    assert (
        _pick_macro_f1(
            {
                "classification_report": {
                    "macro avg": {"f1-score": 0.91, "precision": 0.9, "recall": 0.92}
                }
            }
        )
        == 0.91
    )


def test_pick_macro_f1_missing() -> None:
    assert _pick_macro_f1({}) is None
    assert _pick_macro_f1({"accuracy": 0.99}) is None
