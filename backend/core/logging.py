"""Centralized application logging configuration."""

from __future__ import annotations

import logging

from backend.core.config import settings
from backend.core.log_context import get_correlation_id
from backend.core.log_handlers import LOGGING_CONFIGURED_ATTR, attach_handlers, resolve_log_file_path
from backend.core.log_redaction import RedactingFilter


class CorrelationContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = get_correlation_id() or "-"
        return True


def _build_formatter() -> logging.Formatter:
    if settings.LOG_FORMAT.lower() == "json":
        # Keep text as default; JSON reserved for future structured shipping.
        pass
    return logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(name)s] [correlation_id=%(correlation_id)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def setup_logging(*, force: bool = False) -> None:
    """Configure root logging once for API, workers, and scripts."""
    root_logger = logging.getLogger()
    if getattr(root_logger, LOGGING_CONFIGURED_ATTR, False) and not force:
        root_logger.setLevel(settings.LOG_LEVEL.upper())
        return

    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    root_logger.setLevel(settings.LOG_LEVEL.upper())

    formatter = _build_formatter()
    correlation_filter = CorrelationContextFilter()
    redacting_filter = RedactingFilter()

    log_file = attach_handlers(root_logger)
    for handler in root_logger.handlers:
        handler.setFormatter(formatter)
        handler.addFilter(correlation_filter)
        handler.addFilter(redacting_filter)

    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    startup_logger = logging.getLogger("backend.logging")
    if log_file is not None:
        startup_logger.info(
            "File logging enabled path=%s retention_days=%s",
            log_file,
            settings.LOG_RETENTION_DAYS,
        )
    else:
        startup_logger.info(
            "File logging disabled console=%s level=%s",
            settings.LOG_TO_CONSOLE,
            settings.LOG_LEVEL.upper(),
        )

    setattr(root_logger, LOGGING_CONFIGURED_ATTR, True)
