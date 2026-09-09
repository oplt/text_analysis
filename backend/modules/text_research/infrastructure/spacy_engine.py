"""Optional spaCy linguistic analysis facade (§47).

Install optional NLP extras::

    pip install '.[nlp]'

Then download a language model, e.g. ``python -m spacy download en_core_web_sm``.
Sentence segmentation is available when the loaded model includes a sentencizer
or parser component (standard ``*_core_web_*`` models do).
"""

from __future__ import annotations

from typing import Any

from backend.modules.text_research.infrastructure.linguistic_features import (
    linguistic_features_available,
)


class SpacyLinguisticEngine:
    """Optional spaCy wrapper for POS, dependencies, noun chunks, lemmas, and sentences."""

    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        self.model_name = model_name
        self._nlp: Any = None

    @classmethod
    def available(cls, model_name: str = "en_core_web_sm") -> bool:
        return linguistic_features_available(model_name)

    def _load(self) -> Any:
        if self._nlp is not None:
            return self._nlp
        try:
            import spacy
        except ImportError as exc:
            raise ValueError(
                "SpacyLinguisticEngine requires optional spaCy; "
                "install with: pip install '.[nlp]'"
            ) from exc
        try:
            self._nlp = spacy.load(self.model_name)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"spaCy model {self.model_name!r} unavailable: {exc}") from exc
        return self._nlp

    @staticmethod
    def _sentences(doc: Any) -> list[str]:
        if hasattr(doc, "sents"):
            return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        return []

    def analyze(self, texts: list[str]) -> list[dict[str, Any]]:
        """Return per-text POS, dependency, noun-chunk, lemma, and sentence features."""
        nlp = self._load()
        results: list[dict[str, Any]] = []
        for text in texts:
            doc = nlp(text or "")
            results.append(
                {
                    "pos": [token.pos_ for token in doc if not token.is_space],
                    "deps": [
                        {"token": token.text, "dep": token.dep_, "head": token.head.text}
                        for token in doc
                        if not token.is_space
                    ],
                    "noun_chunks": [chunk.text for chunk in doc.noun_chunks],
                    "lemmas": [token.lemma_ for token in doc if not token.is_space],
                    "sentences": self._sentences(doc),
                }
            )
        return results
