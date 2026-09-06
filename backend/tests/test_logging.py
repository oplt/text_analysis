"""Tests for centralized logging configuration."""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import asyncio

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from starlette.responses import PlainTextResponse

from backend.api.middleware.correlation_id import (
    CORRELATION_ID_HEADER,
    REQUEST_ID_HEADER,
    CorrelationIdMiddleware,
)
from backend.core.config import Settings
from backend.core.log_context import get_correlation_id, reset_correlation_id, set_correlation_id
from backend.core.log_handlers import cleanup_old_logs, resolve_log_file_path
from backend.core.log_redaction import RedactingFilter, redact_message, redact_url
from backend.core.logging import setup_logging


REQUIRED_SETTINGS = {
    "DATABASE_URL": "postgresql+asyncpg://app:app@localhost:5432/app_db",
    "REDIS_URL": "redis://localhost:6379/0",
    "JWT_SECRET": "01234567890123456789012345678901",
    "JWT_ALGORITHM": "HS256",
    "ACCESS_TOKEN_EXPIRE_MINUTES": "15",
    "REFRESH_TOKEN_EXPIRE_DAYS": "7",
}


def test_invalid_log_retention_falls_back_to_default():
    env = {**REQUIRED_SETTINGS, "LOG_RETENTION_DAYS": "0"}
    with patch.dict(os.environ, env, clear=False):
        settings = Settings()
    assert settings.LOG_RETENTION_DAYS == 1


def test_invalid_log_level_falls_back_to_info():
    env = {**REQUIRED_SETTINGS, "LOG_LEVEL": "verbose"}
    with patch.dict(os.environ, env, clear=False):
        settings = Settings()
    assert settings.LOG_LEVEL == "INFO"


def test_resolve_log_file_path_relative_to_backend():
    path = resolve_log_file_path("logs/logs.txt")
    assert str(path).endswith("backend/logs/logs.txt")


def test_cleanup_deletes_only_rotated_application_logs(tmp_path: Path):
    log_file = tmp_path / "logs.txt"
    log_file.write_text("active\n", encoding="utf-8")
    old_date = (datetime.now(UTC).date() - timedelta(days=3)).strftime("%Y-%m-%d")
    rotated = tmp_path / f"logs.txt.{old_date}"
    rotated.write_text("old\n", encoding="utf-8")
    unrelated = tmp_path / "notes.txt"
    unrelated.write_text("keep me\n", encoding="utf-8")

    deleted = cleanup_old_logs(log_file, retention_days=1)

    assert deleted == 1
    assert log_file.exists()
    assert not rotated.exists()
    assert unrelated.exists()


def test_redacting_filter_masks_secrets():
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="authorization=Bearer secret-token password=abc123",
        args=(),
        exc_info=None,
    )
    RedactingFilter().filter(record)
    message = record.getMessage()
    assert "[REDACTED]" in message
    assert "secret-token" not in message
    assert "abc123" not in message


def test_redact_url_masks_credentials():
    redacted = redact_url("postgresql+asyncpg://app:secret@localhost:5432/app_db")
    assert "app:***@" in redacted
    assert "secret" not in redacted


def test_setup_logging_writes_to_file(tmp_path: Path):
    log_file = tmp_path / "logs.txt"
    with patch("backend.core.logging.settings") as mock_settings, patch(
        "backend.core.log_handlers.settings", mock_settings
    ):
        mock_settings.LOG_LEVEL = "INFO"
        mock_settings.LOG_TO_CONSOLE = False
        mock_settings.LOG_TO_FILE = True
        mock_settings.LOG_FILE_PATH = str(log_file)
        mock_settings.LOG_RETENTION_DAYS = 1
        mock_settings.LOG_FORMAT = "text"
        setup_logging(force=True)
        logging.getLogger("backend.test").info("hello from test logger")
        for handler in logging.getLogger().handlers:
            handler.flush()

    assert log_file.exists()
    contents = log_file.read_text(encoding="utf-8")
    assert "hello from test logger" in contents
    assert "correlation_id=-" in contents


def test_correlation_context_is_available_in_logs():
    token = set_correlation_id("req-123")
    try:
        assert get_correlation_id() == "req-123"
    finally:
        reset_correlation_id(token)


def test_redact_message_masks_bearer_token():
    message = redact_message("Authorization: Bearer abc.def.ghi")
    assert "abc.def.ghi" not in message
    assert "[REDACTED]" in message


def test_preserves_incoming_request_id_header():
    async def run() -> None:
        app = FastAPI()
        app.add_middleware(CorrelationIdMiddleware)

        @app.get("/ping")
        async def ping():
            return PlainTextResponse("ok")

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/ping", headers={REQUEST_ID_HEADER: "incoming-id"})

        assert response.status_code == 200
        assert response.headers[CORRELATION_ID_HEADER] == "incoming-id"
        assert response.headers[REQUEST_ID_HEADER] == "incoming-id"

    asyncio.run(run())


def test_generates_request_id_when_missing():
    async def run() -> None:
        app = FastAPI()
        app.add_middleware(CorrelationIdMiddleware)

        @app.get("/ping")
        async def ping():
            return PlainTextResponse("ok")

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/ping")

        assert response.status_code == 200
        assert response.headers[CORRELATION_ID_HEADER]
        assert response.headers[CORRELATION_ID_HEADER] == response.headers[REQUEST_ID_HEADER]

    asyncio.run(run())
