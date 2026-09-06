"""File logging handlers, rotation, and retention cleanup."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from backend.core.config import settings

LOGGING_CONFIGURED_ATTR = "_generic_app_logging_configured"


def resolve_log_file_path(path: str) -> Path:
    log_path = Path(path)
    if not log_path.is_absolute():
        backend_root = Path(__file__).resolve().parents[1]
        log_path = backend_root / log_path
    return log_path


def build_file_handler(log_file: Path) -> TimedRotatingFileHandler:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handler = TimedRotatingFileHandler(
        filename=str(log_file),
        when="midnight",
        interval=1,
        backupCount=0,
        encoding="utf-8",
        utc=True,
    )
    handler.suffix = "%Y-%m-%d"
    return handler


def cleanup_old_logs(log_file: Path, retention_days: int) -> int:
    """Delete rotated log files older than the retention window.

    Only removes files matching ``<log_file.name>.YYYY-MM-DD`` created by our handler.
    Returns the number of deleted files.
    """
    if retention_days < 1:
        retention_days = 1

    log_dir = log_file.parent
    if not log_dir.exists():
        return 0

    cutoff_date = datetime.now(UTC).date() - timedelta(days=retention_days - 1)
    deleted = 0
    prefix = f"{log_file.name}."

    for path in log_dir.iterdir():
        if path == log_file or not path.is_file():
            continue
        if not path.name.startswith(prefix):
            continue

        date_token = path.name[len(prefix) :].split(".", 1)[0]
        try:
            rotated_date = datetime.strptime(date_token, "%Y-%m-%d").date()
        except ValueError:
            continue

        if rotated_date < cutoff_date:
            path.unlink(missing_ok=True)
            deleted += 1

    return deleted


def attach_handlers(root_logger: logging.Logger) -> Path | None:
    log_file: Path | None = None

    if settings.LOG_TO_CONSOLE:
        console = logging.StreamHandler()
        console.set_name("generic_app.console")
        root_logger.addHandler(console)

    if settings.LOG_TO_FILE:
        log_file = resolve_log_file_path(settings.LOG_FILE_PATH)
        file_handler = build_file_handler(log_file)
        file_handler.set_name("generic_app.file")
        root_logger.addHandler(file_handler)
        cleanup_old_logs(log_file, settings.LOG_RETENTION_DAYS)

    return log_file
