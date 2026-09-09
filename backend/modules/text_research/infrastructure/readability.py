"""Readability and style surface metrics (§48).

Pure-Python heuristics — no external readability packages required. These are
descriptive text features, not quality judgments about research content.
"""

from __future__ import annotations

import re
from typing import Any

_SENTENCE_RE = re.compile(r"[.!?]+[\s\"')\]]*|$")
_WORD_RE = re.compile(r"[A-Za-z']+")
_VOWEL_GROUPS = re.compile(r"[aeiouy]+", re.I)


def _syllable_count(word: str) -> int:
    w = word.lower().strip("'")
    if not w:
        return 0
    groups = _VOWEL_GROUPS.findall(w)
    count = len(groups)
    if w.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def readability_metrics(text: str) -> dict[str, Any]:
    """Compute Flesch Reading Ease, Flesch–Kincaid grade, and length stats."""
    words = _WORD_RE.findall(text or "")
    # Rough sentence split: non-empty spans separated by .!?
    raw_sentences = re.split(r"[.!?]+", text or "")
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    n_words = len(words)
    n_sentences = max(len(sentences), 1 if n_words else 0)
    n_syllables = sum(_syllable_count(w) for w in words)
    avg_sentence_len = (n_words / n_sentences) if n_sentences else 0.0
    avg_word_len = (sum(len(w) for w in words) / n_words) if n_words else 0.0
    avg_syllables = (n_syllables / n_words) if n_words else 0.0

    if n_words and n_sentences:
        flesch = 206.835 - 1.015 * avg_sentence_len - 84.6 * avg_syllables
        fk_grade = 0.39 * avg_sentence_len + 11.8 * avg_syllables - 15.59
    else:
        flesch = 0.0
        fk_grade = 0.0

    return {
        "n_words": n_words,
        "n_sentences": n_sentences,
        "n_syllables": n_syllables,
        "avg_sentence_length": avg_sentence_len,
        "avg_word_length": avg_word_len,
        "avg_syllables_per_word": avg_syllables,
        "flesch_reading_ease": flesch,
        "flesch_kincaid_grade": fk_grade,
        "implementation": "text_research.readability.heuristic_v1",
    }


def readability_for_units(
    texts: list[str],
    unit_ids: list[str] | None = None,
) -> dict[str, Any]:
    if unit_ids is not None and len(unit_ids) != len(texts):
        raise ValueError("unit_ids must align 1:1 with texts")
    per_unit = []
    for i, text in enumerate(texts):
        row = readability_metrics(text)
        row["text_unit_id"] = unit_ids[i] if unit_ids else str(i)
        per_unit.append(row)
    corpus_text = "\n\n".join(texts)
    return {
        "corpus": readability_metrics(corpus_text),
        "per_unit": per_unit,
        "n_units": len(texts),
    }
