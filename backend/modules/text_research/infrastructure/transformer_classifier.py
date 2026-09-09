"""Optional transformer-based text classifiers (not part of the core install).

Install optional extras::

    pip install '.[transformers]'
"""

from __future__ import annotations

from typing import Any


def _require_transformers() -> None:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        raise ValueError(
            "Transformer classifiers require optional torch/transformers; "
            "install with: pip install '.[transformers]'"
        ) from exc


def fit_transformer_classifier(
    texts: list[str],
    y: list[Any],
    *,
    model_name: str = "distilbert-base-uncased",
    **kwargs: Any,
) -> dict[str, Any]:
    """Fit a transformer classifier — not implemented in the core platform."""
    _require_transformers()
    raise ValueError(
        "Transformer classifier training is not implemented in the core platform. "
        "Use sparse TF-IDF baselines or embedding_logistic/embedding_svm instead."
    )
