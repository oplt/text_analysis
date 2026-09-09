from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Any

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from backend.core.config import settings

# Primary engine for the API process event loop.
engine = create_async_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    future=True,
)

_default_sessionmaker = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# Worker runtimes own a dedicated event loop. Reusing the API engine there
# attaches asyncpg futures to the wrong loop and poisons the pool.
_sessionmaker_override: ContextVar[async_sessionmaker[AsyncSession] | None] = ContextVar(
    "db_sessionmaker_override",
    default=None,
)


class _SessionLocalFactory:
    """Callable session factory that honors a per-task worker override."""

    def __call__(self, **kwargs: Any) -> AsyncSession:
        maker = _sessionmaker_override.get() or _default_sessionmaker
        return maker(**kwargs)


SessionLocal = _SessionLocalFactory()


def create_worker_sessionmaker() -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    """Create a pooled engine/session maker owned by one worker process loop."""
    worker_engine = create_async_engine(
        settings.DATABASE_URL,
        pool_pre_ping=True,
        future=True,
    )
    maker = async_sessionmaker(
        bind=worker_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    return worker_engine, maker


def install_worker_engine() -> tuple[AsyncEngine, Token]:
    """Bind a worker-local session maker to the current async task.

    Kept for compatibility with callers that need a short-lived isolated
    worker context. Long-running workers use ``create_worker_sessionmaker``
    once per process instead.
    """
    worker_engine, maker = create_worker_sessionmaker()
    token = _sessionmaker_override.set(maker)
    return worker_engine, token


def uninstall_worker_engine(token: Token) -> None:
    _sessionmaker_override.reset(token)
