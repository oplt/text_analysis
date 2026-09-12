"""Bounded multi-query expansion for contradiction / counter-evidence intents."""

from __future__ import annotations

import re
from dataclasses import dataclass

QUERY_EXPANSION_VERSION = "contradiction-variants-v1"

_LANGUAGE_MARKERS = {
    "de": {"der", "die", "das", "und", "von", "für", "widerspricht", "belege"},
    "tr": {"ve", "bu", "şu", "için", "kanıt", "destekleyen", "çelişen"},
    "fr": {"le", "la", "les", "des", "et", "pour", "preuves", "contredisent"},
}
_POLARITY_WORDING = {
    "en": (
        "Evidence supporting or agreeing with",
        "Evidence contradicting, challenging, or presenting counter-arguments to",
    ),
    "de": (
        "Belege, die unterstützen oder zustimmen",
        "Belege, die widersprechen, herausfordern oder Gegenargumente darstellen",
    ),
    "tr": (
        "destekleyen veya aynı fikirde olan kanıtlar",
        "çelişen, karşı çıkan veya karşı argüman sunan kanıtlar",
    ),
    "fr": (
        "éléments qui soutiennent ou confirment",
        "éléments qui contredisent, remettent en cause ou présentent des contre-arguments à",
    ),
}


@dataclass(frozen=True, slots=True)
class QueryVariant:
    label: str
    text: str
    lexical_text: str | None = None
    dense_text: str | None = None
    language: str = "unknown"


class QueryExpansionService:
    """Deterministic, bounded variants — no agent loop."""

    def expand_contradiction(self, query: str) -> list[QueryVariant]:
        q = (query or "").strip()
        if not q:
            return []
        language = _detect_language(q)
        support, contradict = _POLARITY_WORDING[language]
        return [
            QueryVariant(
                label="neutral",
                text=q,
                lexical_text=q,
                dense_text=q,
                language=language,
            ),
            QueryVariant(
                label="support",
                text=f"{support}: {q}",
                lexical_text=q,
                dense_text=f"{support}: {q}",
                language=language,
            ),
            QueryVariant(
                label="contradict",
                text=f"{contradict}: {q}",
                lexical_text=q,
                dense_text=f"{contradict}: {q}",
                language=language,
            ),
        ]

    def expand_if_needed(self, query: str, *, multi_query: bool) -> list[QueryVariant]:
        if not multi_query:
            return [QueryVariant(label="original", text=(query or "").strip())]
        return self.expand_contradiction(query)


def _detect_language(query: str) -> str:
    lowered = query.lower()
    if any(marker in lowered for marker in ("ı", "ğ", "ş")):
        return "tr"
    scores = {
        language: sum(1 for word in re.findall(r"[\wÀ-ÿ]+", lowered) if word in markers)
        for language, markers in _LANGUAGE_MARKERS.items()
    }
    language, score = max(scores.items(), key=lambda item: item[1])
    return language if score else "en"
