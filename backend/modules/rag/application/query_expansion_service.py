"""Bounded multi-query expansion for contradiction / counter-evidence intents."""

from __future__ import annotations

from dataclasses import dataclass

QUERY_EXPANSION_VERSION = "contradiction-variants-v1"


@dataclass(frozen=True, slots=True)
class QueryVariant:
    label: str
    text: str


class QueryExpansionService:
    """Deterministic, bounded variants — no agent loop."""

    def expand_contradiction(self, query: str) -> list[QueryVariant]:
        q = (query or "").strip()
        if not q:
            return []
        return [
            QueryVariant(label="neutral", text=q),
            QueryVariant(
                label="support",
                text=f"Evidence supporting or agreeing with: {q}",
            ),
            QueryVariant(
                label="contradict",
                text=(
                    f"Evidence contradicting, challenging, or presenting "
                    f"counter-arguments to: {q}"
                ),
            ),
        ]

    def expand_if_needed(self, query: str, *, multi_query: bool) -> list[QueryVariant]:
        if not multi_query:
            return [QueryVariant(label="original", text=(query or "").strip())]
        return self.expand_contradiction(query)
