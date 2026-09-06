from __future__ import annotations

from backend.modules.memory.domain.enums import MemoryLevel, MemoryType


class MemoryPolicyService:
    def __init__(self, min_confidence: float):
        self.min_confidence = min_confidence

    def evaluate(
        self,
        *,
        content: str,
        memory_level: MemoryLevel,
        memory_type: MemoryType,
        confidence: float,
        privacy,
    ):
        from backend.modules.memory.domain.policies import evaluate_storage_policy

        return evaluate_storage_policy(
            content=content,
            memory_level=memory_level,
            memory_type=memory_type,
            confidence=confidence,
            privacy=privacy,
            min_confidence=self.min_confidence,
        )
