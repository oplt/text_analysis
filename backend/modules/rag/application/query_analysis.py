"""Deterministic first-pass query analysis for inspectable retrieval profiles."""

from __future__ import annotations

import re
from dataclasses import dataclass

from backend.modules.rag.domain.enums import RetrievalIntent

_QUOTED = re.compile(r"[\"“”'][^\"“”']+[\"“”']")
_COMPARISON = re.compile(
    r"\b(compare|comparison|versus|vs\.?|difference between|vergleich|gegenüber|"
    r"unterschied zwischen|comparer|comparaison|contre|différence entre)\b",
    re.I,
)
_CONTRADICTION = re.compile(
    r"\b(contradict(?:s|ion|ory)?|counter[ -]?evidence|disagree|challenge|oppos(?:e|ing)|"
    r"widerspricht|gegenbeweis|contredit|preuve contraire)\b",
    re.I,
)
_DEFINITION = re.compile(
    r"^(what is|what are|define|definition of|meaning of|was ist|was sind|"
    r"définir|qu'est-ce que)\b",
    re.I,
)
_CITATION_LOOKUP = re.compile(r"\b(citation|cite|cited|source|reference|referenz|quelle)\b", re.I)
_FACT = re.compile(r"^(who|when|where|which year|how many)\b", re.I)
_ENTITY = re.compile(r"\b[A-ZÀ-ÖØ-Þ][\wÀ-ÖØ-öø-ÿ'-]*\b")
_DATE = re.compile(r"\b(?:18|19|20)\d{2}\b")


@dataclass(frozen=True, slots=True)
class QueryAnalysis:
    intent: RetrievalIntent
    reasons: tuple[str, ...]
    lexical_phrase_boost: bool = False
    entity_count: int = 0
    has_date: bool = False

    def to_dict(self) -> dict:
        return {
            "resolved_intent": self.intent.value,
            "reasons": list(self.reasons),
            "lexical_phrase_boost": self.lexical_phrase_boost,
            "entity_count": self.entity_count,
            "has_date": self.has_date,
        }


def analyze_query(query: str) -> QueryAnalysis:
    text = (query or "").strip()
    reasons: list[str] = []
    phrase_boost = bool(_QUOTED.search(text))
    entities = tuple(dict.fromkeys(_ENTITY.findall(text)))
    entity_count = len(entities[1:] if entities and text.startswith(entities[0]) else entities)
    has_date = bool(_DATE.search(text))
    if phrase_boost:
        reasons.append("quoted_phrase")
    if has_date:
        reasons.append("date_marker")
    if _CONTRADICTION.search(text):
        return QueryAnalysis(
            RetrievalIntent.CONTRADICTION,
            tuple([*reasons, "contradiction_marker"]),
            phrase_boost,
            entity_count,
            has_date,
        )
    if _COMPARISON.search(text) and entity_count >= 2:
        return QueryAnalysis(
            RetrievalIntent.COMPARISON,
            tuple([*reasons, "comparison_marker", "two_or_more_entities"]),
            phrase_boost,
            entity_count,
            has_date,
        )
    if _DEFINITION.search(text):
        return QueryAnalysis(
            RetrievalIntent.DEFINITION,
            tuple([*reasons, "definition_pattern"]),
            phrase_boost,
            entity_count,
            has_date,
        )
    if _FACT.search(text):
        return QueryAnalysis(
            RetrievalIntent.FACT,
            tuple([*reasons, "fact_pattern"]),
            phrase_boost,
            entity_count,
            has_date,
        )
    if _CITATION_LOOKUP.search(text):
        reasons.append("citation_lookup")
    return QueryAnalysis(
        RetrievalIntent.EVIDENCE,
        tuple(reasons or ["default_evidence"]),
        phrase_boost,
        entity_count,
        has_date,
    )
