"""Shared helpers for HTTP integration tests."""

from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from typing import Any

# Test-friendly defaults must be set before backend settings are imported.
os.environ.setdefault("REQUIRE_EMAIL_VERIFICATION", "false")
os.environ.setdefault("CELERY_TASK_ALWAYS_EAGER", "true")
os.environ.setdefault("MEMORY_ENABLED", "false")
os.environ.setdefault("RAG_ENABLED", "true")
os.environ.setdefault("CACHE_ENABLED", "false")
os.environ.setdefault("PUBLIC_RATE_LIMIT_REQUESTS", "0")

from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.core.config import settings

_rag_schema_ready: bool | None = None


def integration_enabled() -> bool:
    return os.getenv("RUN_INTEGRATION_TESTS", "").lower() in {"1", "true", "yes"}


def rag_integration_ready() -> bool:
    """True when postgres is migrated through RAG/AI tables."""
    global _rag_schema_ready
    if _rag_schema_ready is not None:
        return _rag_schema_ready
    if not integration_enabled():
        _rag_schema_ready = False
        return _rag_schema_ready

    async def _probe() -> bool:
        await prepare_integration_runtime()
        from backend.db.session import SessionLocal

        try:
            async with SessionLocal() as db:
                await db.execute(text("SELECT 1 FROM rag_documents LIMIT 1"))
                await db.execute(text("SELECT 1 FROM ai_prompt_templates LIMIT 1"))
            return True
        except Exception:
            return False

    import asyncio

    _rag_schema_ready = asyncio.run(_probe())
    return _rag_schema_ready


def unique_email(prefix: str = "integration") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}@example.com"


async def prepare_integration_runtime() -> None:
    """
    Rebind Redis and SQLAlchemy to the current event loop.

    unittest.IsolatedAsyncioTestCase creates a fresh loop per test; the module-level
    clients must be recreated to avoid 'Event loop is closed' failures.
    """
    import redis.asyncio as redis

    from backend.core import cache
    from backend.db import session as db_session

    try:
        await cache.redis_client.aclose()
    except Exception:
        pass
    cache.redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)

    try:
        await db_session.engine.dispose()
    except Exception:
        pass

    db_session.engine = create_async_engine(
        settings.DATABASE_URL,
        pool_pre_ping=True,
        future=True,
    )
    db_session.SessionLocal = async_sessionmaker(
        bind=db_session.engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )


@asynccontextmanager
async def api_client():
    await prepare_integration_runtime()

    from backend.api.main import app

    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            yield client


def csrf_headers(client: AsyncClient) -> dict[str, str]:
    csrf = client.cookies.get(settings.CSRF_COOKIE_NAME)
    if not csrf:
        return {}
    return {settings.CSRF_HEADER_NAME: csrf}


async def register_user(
    client: AsyncClient,
    *,
    email: str | None = None,
    password: str = "Password123!",
    full_name: str = "Integration Test User",
) -> str:
    address = email or unique_email()
    response = await client.post(
        "/api/v1/auth/sign-up",
        json={"email": address, "password": password, "full_name": full_name},
    )
    if response.status_code != 202:
        raise AssertionError(f"sign-up failed: {response.status_code} {response.text}")
    return address


async def sign_in(
    client: AsyncClient,
    *,
    email: str,
    password: str = "Password123!",
) -> None:
    response = await client.post(
        "/api/v1/auth/sign-in",
        json={"email": email, "password": password},
    )
    if response.status_code != 200:
        raise AssertionError(f"sign-in failed: {response.status_code} {response.text}")


async def register_and_sign_in(
    client: AsyncClient,
    *,
    email: str | None = None,
    password: str = "Password123!",
    full_name: str = "Integration Test User",
) -> str:
    address = await register_user(
        client,
        email=email,
        password=password,
        full_name=full_name,
    )
    await sign_in(client, email=address, password=password)
    return address


async def auth_request(client: AsyncClient, method: str, url: str, **kwargs: Any):
    headers = {**csrf_headers(client), **kwargs.pop("headers", {})}
    return await client.request(method, url, headers=headers, **kwargs)
