from __future__ import annotations

import re
from dataclasses import dataclass

from backend.modules.memory.application.memory_router import MemoryRouter, RoutedMemory
from backend.modules.memory.domain.policies import contains_secret, is_ephemeral_statement

EXTRACT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(?i)(user prefers? .{5,120}|prefers? (beginner|detailed|concise).{0,80})"
    ),
    re.compile(r"(?i)(building|working on|creating) (a |an )?.{5,80}"),
    re.compile(r"(?i)(wants?|needs?) .{0,40}(examples?|fastapi|postgres|react)"),
    re.compile(r"(?i)(for project .{3,60},? (the )?(chosen|using|db is) .{3,80})"),
    re.compile(r"(?i)(current plan|next step|working on task)[:.]? .{5,120}"),
)


@dataclass(slots=True)
class ExtractionCandidate:
    content: str
    routed: RoutedMemory
    source_message_id: str | None = None


class MemoryExtractor:
    """Decide what durable facts to store after a conversation turn."""

    def __init__(self, router: MemoryRouter | None = None):
        self.router = router or MemoryRouter()

    def extract_from_turn(
        self,
        *,
        user_message: str,
        assistant_message: str,
        source_message_id: str | None = None,
    ) -> list[ExtractionCandidate]:
        candidates: list[ExtractionCandidate] = []
        for text in (user_message, assistant_message):
            for match in self._iter_sentences(text):
                normalized = match.strip()
                if len(normalized) < 12:
                    continue
                if contains_secret(normalized):
                    continue
                if is_ephemeral_statement(normalized):
                    continue
                if not self._looks_durable(normalized):
                    continue
                routed = self.router.route(content=normalized)
                candidates.append(
                    ExtractionCandidate(
                        content=normalized,
                        routed=routed,
                        source_message_id=source_message_id,
                    )
                )
        return self._dedupe_candidates(candidates)

    def _iter_sentences(self, text: str) -> list[str]:
        parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
        extracted: list[str] = []
        for part in parts:
            part = part.strip()
            if part:
                extracted.append(part)
            for pattern in EXTRACT_PATTERNS:
                for match in pattern.findall(part):
                    if isinstance(match, tuple):
                        extracted.append(" ".join(m for m in match if m))
                    else:
                        extracted.append(str(match))
        return extracted

    def _looks_durable(self, text: str) -> bool:
        if any(pattern.search(text) for pattern in EXTRACT_PATTERNS):
            return True
        durable_markers = (
            "prefer",
            "building",
            "project",
            "chosen",
            "using",
            "goal",
            "always",
            "instruction",
            "postgres",
            "fastapi",
        )
        lowered = text.lower()
        return any(marker in lowered for marker in durable_markers)

    def _dedupe_candidates(
        self, candidates: list[ExtractionCandidate]
    ) -> list[ExtractionCandidate]:
        seen: set[str] = set()
        unique: list[ExtractionCandidate] = []
        for candidate in candidates:
            key = candidate.content.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(candidate)
        return unique
