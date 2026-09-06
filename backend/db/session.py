from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Any

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

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

# Eager/Celery sync workers call asyncio.run() on a fresh loop. Reusing the API
# engine there attaches asyncpg futures to the wrong loop and poisons the pool.
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


def install_worker_engine() -> tuple[AsyncEngine, Token]:
    """Bind a NullPool engine to the current asyncio task for worker jobs."""
    worker_engine = create_async_engine(
        settings.DATABASE_URL,
        pool_pre_ping=True,
        poolclass=NullPool,
        future=True,
    )
    maker = async_sessionmaker(
        bind=worker_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    token = _sessionmaker_override.set(maker)
    return worker_engine, token


def uninstall_worker_engine(token: Token) -> None:
    _sessionmaker_override.reset(token)
