"""Redact sensitive values from log messages."""

from __future__ import annotations

import logging
import re
from urllib.parse import urlsplit, urlunsplit

def redact_message(message: str) -> str:
    redacted = re.sub(r"(?i)\bbearer\s+\S+", "Bearer [REDACTED]", message)
    redacted = re.sub(
        r"(?i)(password|secret|token|api[_-]?key|authorization)\s*[:=]\s*\S+",
        r"\1=[REDACTED]",
        redacted,
    )
    return redacted


def redact_url(url: str) -> str:
    parts = urlsplit(url)
    if not parts.netloc or "@" not in parts.netloc:
        return url
    credentials, host = parts.netloc.rsplit("@", 1)
    if ":" in credentials:
        username = credentials.split(":", 1)[0]
        netloc = f"{username}:***@{host}"
    else:
        netloc = f"***@{host}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        record.msg = redact_message(message)
        record.args = ()
        return True
