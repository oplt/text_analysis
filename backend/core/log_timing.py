"""Timing helpers for structured latency logging."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any

from backend.core.config import settings


def _format_fields(fields: dict[str, Any]) -> str:
    parts: list[str] = []
    for key, value in fields.items():
        if value is None:
            continue
        parts.append(f"{key}={value}")
    return " ".join(parts)


@contextmanager
def timed_operation(
    logger: logging.Logger,
    operation: str,
    *,
    slow_ms: int | None = None,
    level: int = logging.INFO,
    **fields: Any,
) -> Iterator[None]:
    started = perf_counter()
    suffix = _format_fields(fields)
    try:
        yield
    except Exception:
        duration_ms = (perf_counter() - started) * 1000
        logger.exception(
            "%s failed duration_ms=%.2f %s",
            operation,
            duration_ms,
            suffix,
        )
        raise
    else:
        duration_ms = (perf_counter() - started) * 1000
        threshold = slow_ms if slow_ms is not None else settings.SLOW_EXTERNAL_CALL_MS
        message = f"{operation} completed duration_ms={duration_ms:.2f} {suffix}".strip()
        if duration_ms >= threshold:
            logger.warning("slow_operation %s", message)
        else:
            logger.log(level, message)
