"""Lightweight language detection and routing for text research.

Primary path: heuristic unicode-script + stopword-overlap detection for
``en``, ``de``, ``fr``, ``tr`` with ``und`` fallback. Optional ``langdetect``
is used when installed; it is not a hard dependency.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from backend.modules.text_research.infrastructure.language_processing import stopwords_for

DETECTOR_VERSION = "1"
_SUPPORTED_HEURISTIC_LANGS = ("en", "de", "fr", "tr")
_WORD_RE = re.compile(r"[\w']+", re.UNICODE)


@runtime_checkable
class LanguageDetector(Protocol):
    """Protocol for text-language detectors."""

    def detect(self, text: str) -> dict[str, Any]: ...


def _tokenize_words(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text) if t.strip()]


def _script_counts(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ch in text:
        if ch.isspace() or not ch.isalpha():
            continue
        script = unicodedata.name(ch, "").split()[0] if unicodedata.name(ch, "") else "OTHER"
        counts[script] = counts.get(script, 0) + 1
    return counts


@dataclass
class HeuristicLanguageDetector:
    """Unicode-script and stopword-overlap language detector."""

    name: str = "heuristic"
    version: str = DETECTOR_VERSION

    def detect(self, text: str) -> dict[str, Any]:
        words = _tokenize_words(text)
        if not words:
            return {
                "language": "und",
                "confidence": 0.0,
                "detector_name": self.name,
                "detector_version": self.version,
            }

        scores: dict[str, float] = {}
        for lang in _SUPPORTED_HEURISTIC_LANGS:
            stopwords = stopwords_for(lang)
            if not stopwords:
                scores[lang] = 0.0
                continue
            overlap = sum(1 for w in words if w in stopwords)
            scores[lang] = overlap / len(words)

        best_lang = max(scores, key=scores.get)
        best_score = scores[best_lang]

        scripts = _script_counts(text)
        latin_ratio = scripts.get("LATIN", 0) / max(sum(scripts.values()), 1)
        if best_score < 0.02 and latin_ratio < 0.5:
            return {
                "language": "und",
                "confidence": float(best_score),
                "detector_name": self.name,
                "detector_version": self.version,
                "scores": scores,
            }

        if best_score <= 0.0:
            language = "und"
            confidence = 0.0
        else:
            language = best_lang
            confidence = float(min(1.0, best_score * 4.0))

        return {
            "language": language,
            "confidence": confidence,
            "detector_name": self.name,
            "detector_version": self.version,
            "scores": scores,
        }


@dataclass
class LangdetectLanguageDetector:
    """Optional ``langdetect`` wrapper (only when the package is installed)."""

    name: str = "langdetect"
    version: str = DETECTOR_VERSION

    def detect(self, text: str) -> dict[str, Any]:
        try:
            import langdetect
        except ImportError as exc:
            raise RuntimeError("langdetect is not installed") from exc

        try:
            detected = langdetect.detect_langs(text)
        except langdetect.lang_detect_exception.LangDetectException:
            return {
                "language": "und",
                "confidence": 0.0,
                "detector_name": self.name,
                "detector_version": self.version,
            }

        if not detected:
            return {
                "language": "und",
                "confidence": 0.0,
                "detector_name": self.name,
                "detector_version": self.version,
            }

        top = detected[0]
        return {
            "language": top.lang,
            "confidence": float(top.prob),
            "detector_name": self.name,
            "detector_version": self.version,
            "candidates": [{"language": item.lang, "confidence": float(item.prob)} for item in detected],
        }


class LanguageRouter:
    """Map detected language codes to preprocessing processor profiles."""

    _PROCESSORS = {
        "en": "en",
        "de": "de",
        "fr": "fr",
        "tr": "tr",
    }

    def processor_for(self, lang: str | None) -> str:
        if not lang:
            return "generic"
        code = str(lang).strip().lower().split("-")[0]
        return self._PROCESSORS.get(code, "generic")

    @staticmethod
    def for_language(lang: str | None) -> str:
        return LanguageRouter().processor_for(lang)


def default_language_detector() -> LanguageDetector:
    """Return the best available detector (langdetect when installed, else heuristic)."""
    try:
        import langdetect  # noqa: F401

        return LangdetectLanguageDetector()
    except ImportError:
        return HeuristicLanguageDetector()


def detect_units(
    texts: list[str],
    *,
    manual_overrides: dict[str, str] | None = None,
    unit_ids: list[str] | None = None,
    detector: LanguageDetector | None = None,
) -> list[dict[str, Any]]:
    """Detect language independently for each text unit.

    ``manual_overrides`` maps unit id (or ``str(index)``) to a language code;
    overridden units receive confidence 1.0 and ``detector_name=manual_override``.
    """
    active = detector or default_language_detector()
    overrides = manual_overrides or {}
    resolved_ids = unit_ids or [f"unit-{index}" for index in range(len(texts))]
    if len(resolved_ids) != len(texts):
        raise ValueError("unit_ids must align 1:1 with texts")

    results: list[dict[str, Any]] = []
    for index, (unit_id, text) in enumerate(zip(resolved_ids, texts, strict=True)):
        override = overrides.get(unit_id) or overrides.get(str(index))
        result = detect_with_override(text, manual_override=override, detector=active)
        result["unit_id"] = unit_id
        result["unit_index"] = index
        results.append(result)
    return results


def detect_with_override(
    text: str,
    manual_override: str | None = None,
    *,
    detector: LanguageDetector | None = None,
) -> dict[str, Any]:
    """Detect language, honoring an explicit manual override when provided."""
    if manual_override:
        code = str(manual_override).strip().lower().split("-")[0]
        return {
            "language": code,
            "confidence": 1.0,
            "detector_name": "manual_override",
            "detector_version": DETECTOR_VERSION,
            "overridden": True,
        }

    active = detector or HeuristicLanguageDetector()
    result = active.detect(text)
    result["overridden"] = False
    return result
