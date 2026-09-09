"""Domain-aware scientific warnings for research UX (Phase 15).

These are interpretive review signals — not automatic lifecycle decisions.
Feature-selection leakage is never soft-warned here: it is prevented in
``classifiers._fit_classifier_and_evaluate`` and covered by hard tests.
"""

from __future__ import annotations

from typing import Any


def format_warning(message: str) -> str:
    text = message.strip()
    if not text:
        return text
    if text.lower().startswith("warning:"):
        return text
    return f"Warning: {text}"


def ci_width(ci: dict[str, Any] | None) -> float | None:
    if not isinstance(ci, dict):
        return None
    lower = ci.get("lower")
    upper = ci.get("upper")
    if lower is None:
        lower = ci.get("low")
    if upper is None:
        upper = ci.get("high")
    try:
        return float(upper) - float(lower)
    except (TypeError, ValueError):
        return None


def annotation_category_prevalence(value_matrix: list[list[Any]]) -> dict[str, float]:
    """Share of non-null annotation cells by category label."""
    counts: dict[str, int] = {}
    total = 0
    for row in value_matrix:
        for value in row:
            if value is None:
                continue
            key = str(value)
            counts[key] = counts.get(key, 0) + 1
            total += 1
    if total <= 0:
        return {}
    return {key: count / total for key, count in counts.items()}


def dominant_category_share(prevalence: dict[str, float]) -> float | None:
    if not prevalence:
        return None
    return max(prevalence.values())


def dfm_scientific_warnings(
    *,
    unit_count: int,
    feature_count: int,
    nnz: int,
    density: float,
    estimated_memory_bytes: int | None = None,
) -> list[str]:
    warnings: list[str] = []
    sparsity = max(0.0, min(1.0, 1.0 - float(density)))
    if unit_count <= 0 or feature_count <= 0:
        warnings.append("DFM is empty — no documents or features were retained.")
    if unit_count > 0 and unit_count < 20:
        warnings.append(
            f"Only {unit_count} DFM documents are available; sparsity and keyness "
            "estimates may be unstable."
        )
    if feature_count > 0 and feature_count < 10:
        warnings.append(
            f"Only {feature_count} features remain after pruning; interpretive "
            "comparisons may be brittle."
        )
    if sparsity >= 0.995 and nnz > 0:
        warnings.append(
            f"DFM sparsity is {sparsity:.1%} ({nnz} non-zero cells) — memory and "
            "association stats need care."
        )
    if estimated_memory_bytes is not None and estimated_memory_bytes >= 512 * 1024 * 1024:
        mb = estimated_memory_bytes / (1024 * 1024)
        warnings.append(
            f"Estimated dense-equivalent DFM memory is ~{mb:.0f} MiB; prefer sparse storage."
        )
    return [format_warning(message) for message in warnings]


def estimate_dfm_memory_bytes(*, unit_count: int, feature_count: int, nnz: int) -> dict[str, int]:
    """Rough memory footprints for audit panels (float64 dense vs COO sparse)."""
    dense = max(0, int(unit_count) * int(feature_count) * 8)
    # COO: row/col int32 + data float64 ≈ 4+4+8 per nnz, plus feature name overhead ignored
    sparse = max(0, int(nnz) * 16)
    return {
        "dense_float64_bytes": dense,
        "sparse_coo_bytes": sparse,
        "preferred_bytes": sparse if sparse and sparse < dense else dense,
    }


def classifier_scientific_warnings(
    *,
    feature_space: dict[str, Any] | None,
    n_train: int | None = None,
    n_test: int | None = None,
    class_prevalence: dict[str, float] | None = None,
) -> list[str]:
    """Warnings for classifier feature space / sample size — never leakage soft-warnings."""
    warnings: list[str] = []
    space = feature_space or {}
    raw = space.get("raw_vocabulary")
    after_df = space.get("after_df_pruning")
    after_sel = space.get("after_supervised_selection")
    selection = str(space.get("selection_step") or "none")

    if isinstance(raw, int) and isinstance(after_df, int):
        warnings.append(
            f"Classifier vocabulary: raw={raw}, post-min_df={after_df}, "
            f"selected={after_sel if after_sel is not None else after_df} "
            f"(selection={selection})."
        )
    if (
        isinstance(after_sel, int)
        and isinstance(n_train, int)
        and n_train > 0
        and after_sel > max(50, n_train * 20)
    ):
        warnings.append(
            f"Selected feature count ({after_sel}) is very large relative to train "
            f"size ({n_train})."
        )
    if isinstance(after_sel, int) and after_sel == 0:
        warnings.append("Supervised feature selection retained zero features.")
    if isinstance(n_train, int) and 0 < n_train < 30:
        warnings.append(
            f"Only {n_train} training units — holdout metrics and feature selection "
            "may be unstable."
        )
    if isinstance(n_test, int) and 0 < n_test < 20:
        warnings.append(f"Only {n_test} test units — reported metrics have high sampling variance.")

    share = dominant_category_share(class_prevalence or {})
    if share is not None and share >= 0.85:
        warnings.append(
            f"{share:.0%} of training labels use one category. Metrics may look optimistic."
        )

    # Explicit non-warning: leakage is prevented in fit path; do not emit soft leakage text.
    return [format_warning(message) for message in warnings]


def assert_no_leakage_soft_warning(messages: list[str]) -> None:
    """Guard: soft UX must never claim leakage that the pipeline forbids."""
    needle = "feature selection was fitted outside the training fold"
    for message in messages:
        if needle in message.lower():
            raise AssertionError(
                "Leakage soft-warning is forbidden; pipeline must prevent out-of-fold selection fit"
            )
