from __future__ import annotations

import re
from dataclasses import dataclass

from backend.modules.memory.domain.enums import MemoryLevel, MemorySource, MemoryType

TASK_STATE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\b(current(ly)?|working on|in progress|next step|todo|task)\b"),
    re.compile(r"(?i)\b(plan|step \d+|unfinished|pending)\b"),
)

PROJECT_DECISION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\b(chosen|selected|using|decided|architecture|stack)\b"),
    re.compile(r"(?i)\b(postgres|postgresql|mysql|mongodb|redis|fastapi|django)\b"),
)

PREFERENCE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\b(prefer|likes?|wants?|always use|please use)\b"),
    re.compile(r"(?i)\b(beginner.friendly|detailed|concise|examples?)\b"),
)

EVENT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\b(completed|finished|started|changed|corrected|onboarding)\b"),
)

AGENT_RULE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\b(always|never|must|should always|tool preference)\b"),
)


@dataclass(slots=True)
class RoutedMemory:
    content: str
    memory_level: MemoryLevel
    memory_type: MemoryType
    confidence: float
    source: MemorySource = MemorySource.CHAT


class MemoryRouter:
    """Route candidate facts to the correct memory layer."""

    def route(
        self,
        *,
        content: str,
        explicit_level: MemoryLevel | None = None,
        explicit_type: MemoryType | None = None,
        is_task_state: bool = False,
        is_project_decision: bool = False,
        is_agent_rule: bool = False,
        is_event: bool = False,
    ) -> RoutedMemory:
        if explicit_level and explicit_type:
            return RoutedMemory(
                content=content,
                memory_level=explicit_level,
                memory_type=explicit_type,
                confidence=0.9,
            )

        if is_task_state or any(p.search(content) for p in TASK_STATE_PATTERNS):
            return RoutedMemory(
                content=content,
                memory_level=MemoryLevel.SESSION,
                memory_type=MemoryType.TASK_STATE,
                confidence=0.7,
            )

        if is_project_decision or any(p.search(content) for p in PROJECT_DECISION_PATTERNS):
            return RoutedMemory(
                content=content,
                memory_level=MemoryLevel.PROJECT,
                memory_type=MemoryType.DECISION,
                confidence=0.82,
            )

        if is_agent_rule or any(p.search(content) for p in AGENT_RULE_PATTERNS):
            return RoutedMemory(
                content=content,
                memory_level=MemoryLevel.AGENT,
                memory_type=MemoryType.INSTRUCTION,
                confidence=0.78,
            )

        if is_event or any(p.search(content) for p in EVENT_PATTERNS):
            return RoutedMemory(
                content=content,
                memory_level=MemoryLevel.EPISODIC,
                memory_type=MemoryType.EVENT,
                confidence=0.75,
            )

        if any(p.search(content) for p in PREFERENCE_PATTERNS):
            return RoutedMemory(
                content=content,
                memory_level=MemoryLevel.USER,
                memory_type=MemoryType.PREFERENCE,
                confidence=0.85,
            )

        return RoutedMemory(
            content=content,
            memory_level=MemoryLevel.USER,
            memory_type=MemoryType.FACT,
            confidence=0.72,
        )
