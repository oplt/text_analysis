from __future__ import annotations

from datetime import UTC, datetime


def memory_recency_weight(timestamp: datetime | None) -> float:
    if timestamp is None:
        return 0.0

    seen = timestamp
    if seen.tzinfo is None:
        seen = seen.replace(tzinfo=UTC)

    now = datetime.now(UTC)
    age_days = max(0.0, (now - seen).total_seconds() / 86400.0)
    if age_days <= 0:
        return 1.0
    return max(0.0, 1.0 - (age_days / 90.0))
