"""Analysis-specific scientific applicability policy (TASK-018).

Produces warnings/errors for known incompatible analysis + preprocessing
combinations. Does not silently rewrite user profiles.
"""

from __future__ import annotations

from typing import Any

from backend.modules.text_research.infrastructure.scientific_warnings import format_warning

# Flesch / Flesch–Kincaid heuristics assume English syllable patterns.
READABILITY_SUPPORTED_LANGUAGES = frozenset({"en", "eng", "english"})


def _normalize_language(value: str | None) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def readability_language_warnings(*, language: str | None) -> list[str]:
    code = _normalize_language(language)
    if not code:
        return [
            format_warning(
                "Readability metrics (Flesch / Flesch–Kincaid) assume English orthography; "
                "no language was specified."
            )
        ]
    short = code.split("-", 1)[0]
    if short in READABILITY_SUPPORTED_LANGUAGES or code in READABILITY_SUPPORTED_LANGUAGES:
        return []
    return [
        format_warning(
            f"Readability metrics are calibrated for English; language={language!r} "
            "may yield misleading Flesch / grade-level scores."
        )
    ]


def kwic_case_sensitivity_warnings(
    *,
    case_sensitive: bool,
    preprocessing: dict[str, Any] | None,
) -> list[str]:
    if not case_sensitive:
        return []
    profile = preprocessing or {}
    if profile.get("lowercase"):
        return [
            format_warning(
                "KWIC case_sensitive=true conflicts with a lowercase preprocessing profile; "
                "matches will not distinguish case after tokenization."
            )
        ]
    return []


def phrase_morphology_warnings(
    *,
    analysis_type: str,
    preprocessing: dict[str, Any] | None,
) -> list[str]:
    profile = preprocessing or {}
    if analysis_type not in {"ngrams", "cooccurrence", "dictionary", "frequencies"}:
        return []
    warnings: list[str] = []
    if profile.get("stemming") and profile.get("lemmatization"):
        warnings.append(
            format_warning(
                "Both stemming and lemmatization are enabled; phrase/collocation identity "
                "may be unstable across morphology settings."
            )
        )
    if analysis_type in {"ngrams", "cooccurrence"} and profile.get("remove_stopwords") is False:
        warnings.append(
            format_warning(
                f"{analysis_type} retains stopwords; association scores may be dominated by "
                "function words unless that is intentional."
            )
        )
    return warnings


def validate_analysis_policy(
    *,
    analysis_type: str,
    language: str | None = None,
    case_sensitive: bool | None = None,
    preprocessing: dict[str, Any] | None = None,
) -> list[str]:
    """Return applicability warnings for the given analysis configuration."""
    out: list[str] = []
    if analysis_type == "readability":
        out.extend(readability_language_warnings(language=language))
    if analysis_type == "kwic" and case_sensitive is not None:
        out.extend(
            kwic_case_sensitivity_warnings(
                case_sensitive=case_sensitive,
                preprocessing=preprocessing,
            )
        )
    out.extend(
        phrase_morphology_warnings(
            analysis_type=analysis_type,
            preprocessing=preprocessing,
        )
    )
    return out
