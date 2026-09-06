from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from backend.modules.memory.domain.enums import MemoryLevel, MemoryPrivacy, MemoryType

SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\b(password|passwd|pwd)\s*[:=]\s*\S+"),
    re.compile(r"(?i)\b(api[_-]?key|apikey)\s*[:=]\s*\S+"),
    re.compile(r"(?i)\b(secret|token|bearer)\s*[:=]\s*\S+"),
    re.compile(r"(?i)\b(sk-[a-zA-Z0-9]{10,})\b"),
    re.compile(r"(?i)\b(ghp_[a-zA-Z0-9]{20,})\b"),
    re.compile(r"(?i)\b(eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,})\b"),
    re.compile(r"(?i)\b(-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----)"),
    re.compile(r"(?i)\b(credit\s*card|cvv|cvc)\s*[:=]?\s*\d"),
    re.compile(r"(?i)\b(ssn|social security)\s*[:=]?\s*\d"),
)

EPHEMERAL_PHRASES: tuple[str, ...] = (
    "i'm tired",
    "i am tired",
    "feeling sad",
    "just kidding",
    "never mind",
    "nvm",
)

LONG_TERM_TYPES: frozenset[MemoryType] = frozenset(
    {
        MemoryType.PREFERENCE,
        MemoryType.FACT,
        MemoryType.GOAL,
        MemoryType.SKILL_LEVEL,
        MemoryType.INSTRUCTION,
        MemoryType.WARNING,
        MemoryType.DECISION,
        MemoryType.PROJECT_CONTEXT,
    }
)


@dataclass(slots=True)
class PolicyDecision:
    allowed: bool
    reason: str | None = None
    adjusted_confidence: float | None = None
    privacy: MemoryPrivacy = MemoryPrivacy.NORMAL


def contains_secret(content: str) -> bool:
    return any(pattern.search(content) for pattern in SECRET_PATTERNS)


def is_ephemeral_statement(content: str) -> bool:
    normalized = content.strip().lower()
    return any(phrase in normalized for phrase in EPHEMERAL_PHRASES)


def evaluate_storage_policy(
    *,
    content: str,
    memory_level: MemoryLevel,
    memory_type: MemoryType,
    confidence: float,
    privacy: MemoryPrivacy,
    min_confidence: float,
) -> PolicyDecision:
    if privacy == MemoryPrivacy.DO_NOT_STORE:
        return PolicyDecision(allowed=False, reason="privacy_do_not_store")

    if contains_secret(content):
        return PolicyDecision(allowed=False, reason="secret_detected")

    if is_ephemeral_statement(content) and memory_level != MemoryLevel.SESSION:
        return PolicyDecision(allowed=False, reason="ephemeral_statement")

    if memory_level == MemoryLevel.USER and memory_type not in LONG_TERM_TYPES:
        return PolicyDecision(
            allowed=False,
            reason="invalid_user_memory_type",
        )

    if confidence < min_confidence and memory_level in {
        MemoryLevel.USER,
        MemoryLevel.AGENT,
        MemoryLevel.PROJECT,
    }:
        return PolicyDecision(allowed=False, reason="confidence_below_threshold")

    adjusted = confidence
    if privacy == MemoryPrivacy.SENSITIVE and memory_level == MemoryLevel.USER:
        adjusted = min(confidence, 0.75)

    return PolicyDecision(allowed=True, adjusted_confidence=adjusted, privacy=privacy)


def session_expires_at(ttl_days: int) -> datetime:
    return datetime.now(UTC) + timedelta(days=ttl_days)
