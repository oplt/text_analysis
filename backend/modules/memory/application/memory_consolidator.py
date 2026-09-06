from __future__ import annotations

from backend.modules.memory.domain.models import MemoryItem


class MemoryConsolidator:
    """Merge duplicates and prefer newer confirmed facts."""

    def consolidate(self, items: list[MemoryItem]) -> list[MemoryItem]:
        if not items:
            return []

        grouped: dict[str, MemoryItem] = {}
        for item in items:
            key = self._similarity_key(item.content)
            existing = grouped.get(key)
            if existing is None:
                grouped[key] = item
                continue
            grouped[key] = self._pick_preferred(existing, item)
        return list(grouped.values())

    def mark_contradictions(self, items: list[MemoryItem]) -> list[MemoryItem]:
        """Drop older contradicted items when a newer confirmed fact exists."""
        by_type: dict[str, list[MemoryItem]] = {}
        for item in items:
            key = f"{item.metadata.memory_type.value}:{item.metadata.memory_level.value}"
            by_type.setdefault(key, []).append(item)

        result: list[MemoryItem] = []
        for group in by_type.values():
            if len(group) == 1:
                result.append(group[0])
                continue
            group.sort(key=self._recency_score, reverse=True)
            winner = group[0]
            for challenger in group[1:]:
                if self._contradicts(winner.content, challenger.content):
                    continue
                result.append(challenger)
            result.append(winner)
        return result

    def _similarity_key(self, content: str) -> str:
        normalized = " ".join(content.lower().split())
        return normalized[:160]

    def _recency_score(self, item: MemoryItem) -> float:
        confirmed = item.metadata.last_confirmed_at or item.metadata.created_at
        seen = item.metadata.last_seen_at or confirmed
        base = 0.0
        if confirmed:
            base += confirmed.timestamp()
        if seen:
            base += seen.timestamp() * 0.001
        if item.metadata.confidence:
            base += item.metadata.confidence
        return base

    def _pick_preferred(self, left: MemoryItem, right: MemoryItem) -> MemoryItem:
        return left if self._recency_score(left) >= self._recency_score(right) else right

    def _contradicts(self, left: str, right: str) -> bool:
        left_l = left.lower()
        right_l = right.lower()
        negations = ("not ", "no longer ", "don't ", "do not ")
        for neg in negations:
            if neg in left_l and right_l.replace(neg, "") in left_l:
                return True
            if neg in right_l and left_l.replace(neg, "") in right_l:
                return True
        return False
