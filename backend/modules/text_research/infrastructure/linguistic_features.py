"""POS / linguistic feature extraction (§47).

Optional spaCy-backed POS histograms. Lemma support already exists via
preprocessing/simplemma; this module does not claim POS when spaCy is absent.
"""

from __future__ import annotations

from collections import Counter
from typing import Any


def linguistic_features_available(model_name: str = "en_core_web_sm") -> bool:
    try:
        import spacy

        spacy.load(model_name)
        return True
    except Exception:  # noqa: BLE001
        return False


def pos_tag_histogram(
    texts: list[str],
    *,
    model_name: str = "en_core_web_sm",
    unit_ids: list[str] | None = None,
) -> dict[str, Any]:
    if unit_ids is not None and len(unit_ids) != len(texts):
        raise ValueError("unit_ids must align 1:1 with texts")
    try:
        import spacy
    except ImportError as exc:
        raise ValueError(
            "POS tagging requires optional spaCy; not installed in this environment."
        ) from exc
    try:
        nlp = spacy.load(model_name)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"spaCy model {model_name!r} unavailable: {exc}") from exc

    corpus_counts: Counter[str] = Counter()
    per_unit: list[dict[str, Any]] = []
    for i, text in enumerate(texts):
        doc = nlp(text or "")
        counts: Counter[str] = Counter(tok.pos_ for tok in doc if not tok.is_space)
        corpus_counts.update(counts)
        per_unit.append(
            {
                "text_unit_id": unit_ids[i] if unit_ids else str(i),
                "pos_counts": dict(counts),
                "n_tokens": sum(counts.values()),
            }
        )
    return {
        "available": True,
        "model": model_name,
        "corpus_pos_counts": dict(corpus_counts),
        "per_unit": per_unit,
    }


def describe_linguistic_capabilities() -> dict[str, Any]:
    from backend.modules.text_research.infrastructure.spacy_engine import SpacyLinguisticEngine

    return {
        "pos_available": linguistic_features_available(),
        "spacy_engine_available": SpacyLinguisticEngine.available(),
        "lemma_via_preprocessing": True,
        "dependency_frequencies": SpacyLinguisticEngine.available(),
        "notes": [
            "POS/dependency require optional spaCy (see SpacyLinguisticEngine).",
            "Lemmatization available via preprocessing when simplemma supports the language.",
        ],
    }
