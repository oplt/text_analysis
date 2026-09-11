from __future__ import annotations

from contextvars import ContextVar, Token
from time import monotonic
from typing import Any, Literal

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from backend.core.config import settings
from backend.db.pool_observability import instrument_engine

DatabaseRole = Literal["api", "worker"]


def database_engine_kwargs(*, role: DatabaseRole = "api") -> dict[str, Any]:
    """Explicit SQLAlchemy pool settings for the API or a Celery worker process.

    Workers use a smaller pool because Celery prefork multiplies connections by
    ``--concurrency``. Never rely on SQLAlchemy silent defaults in production.
    """
    if role == "worker":
        return {
            "pool_size": max(1, int(settings.DB_WORKER_POOL_SIZE)),
            "max_overflow": max(0, int(settings.DB_WORKER_MAX_OVERFLOW)),
            "pool_timeout": max(1, int(settings.DB_WORKER_POOL_TIMEOUT)),
            "pool_recycle": max(1, int(settings.DB_WORKER_POOL_RECYCLE)),
            "pool_pre_ping": True,
            "future": True,
        }
    return {
        "pool_size": max(1, int(settings.DB_POOL_SIZE)),
        "max_overflow": max(0, int(settings.DB_MAX_OVERFLOW)),
        "pool_timeout": max(1, int(settings.DB_POOL_TIMEOUT)),
        "pool_recycle": max(1, int(settings.DB_POOL_RECYCLE)),
        "pool_pre_ping": True,
        "future": True,
    }


def describe_database_pool_policy() -> dict[str, Any]:
    """Ops snapshot: API vs worker pool math + PgBouncer guidance."""
    api = database_engine_kwargs(role="api")
    worker = database_engine_kwargs(role="worker")
    return {
        "api": {
            "pool_size": api["pool_size"],
            "max_overflow": api["max_overflow"],
            "pool_timeout": api["pool_timeout"],
            "pool_recycle": api["pool_recycle"],
            "max_connections_per_api_process": api["pool_size"] + api["max_overflow"],
        },
        "worker": {
            "pool_size": worker["pool_size"],
            "max_overflow": worker["max_overflow"],
            "pool_timeout": worker["pool_timeout"],
            "pool_recycle": worker["pool_recycle"],
            "max_connections_per_worker_process": worker["pool_size"] + worker["max_overflow"],
            "note": (
                "Multiply by Celery --concurrency (and by number of worker "
                "deployments) when sizing Postgres / PgBouncer."
            ),
        },
        "settings": {
            "DB_POOL_SIZE": settings.DB_POOL_SIZE,
            "DB_MAX_OVERFLOW": settings.DB_MAX_OVERFLOW,
            "DB_POOL_TIMEOUT": settings.DB_POOL_TIMEOUT,
            "DB_POOL_RECYCLE": settings.DB_POOL_RECYCLE,
            "DB_WORKER_POOL_SIZE": settings.DB_WORKER_POOL_SIZE,
            "DB_WORKER_MAX_OVERFLOW": settings.DB_WORKER_MAX_OVERFLOW,
            "DB_WORKER_POOL_TIMEOUT": settings.DB_WORKER_POOL_TIMEOUT,
            "DB_WORKER_POOL_RECYCLE": settings.DB_WORKER_POOL_RECYCLE,
        },
        "pgbouncer": {
            "recommended": True,
            "pool_mode": "transaction",
            "runbook": "docs/runbooks/postgres-pgbouncer.md",
            "guidance": (
                "Put PgBouncer between app processes and Postgres. Use transaction "
                "pooling for asyncpg/SQLAlchemy request/worker short sessions. Size "
                "PgBouncer default_pool_size from expected concurrent clients, not "
                "from SQLAlchemy pool_size alone."
            ),
        },
        "connection_budget": {
            "formula": (
                "api_replicas * api_processes * (DB_POOL_SIZE + DB_MAX_OVERFLOW) + "
                "celery_deployments * celery_concurrency * "
                "(DB_WORKER_POOL_SIZE + DB_WORKER_MAX_OVERFLOW) + admin "
                "< postgres_or_pgbouncer_capacity"
            ),
            "do_not_blindly_increase_pools": True,
        },
    }


# Primary engine for the API process event loop.
engine = create_async_engine(settings.DATABASE_URL, **database_engine_kwargs(role="api"))
instrument_engine(role="api", sync_engine=engine.sync_engine)


def observe_session_duration(*, role: DatabaseRole, started_at: float) -> None:
    """Record session lifetime at dependency/task scope boundaries."""
    try:
        from backend.observability.prometheus_metrics import database_session_duration_seconds

        database_session_duration_seconds.labels(role=role).observe(monotonic() - started_at)
    except Exception:
        pass


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


@event.listens_for(AsyncSession.sync_session_class, "after_commit")
def _publish_research_run_events_after_commit(session: Any) -> None:
    """Publish run events only after their transaction is durable."""
    events = session.info.pop("research_run_events", [])
    if not events:
        return
    from backend.modules.text_research.infrastructure.run_events import (
        _schedule_run_event_publish,
    )

    for envelope in events:
        _schedule_run_event_publish(envelope)


@event.listens_for(AsyncSession.sync_session_class, "after_rollback")
def _discard_research_run_events_after_rollback(session: Any) -> None:
    """Rolled-back state is never eligible for SSE publication."""
    session.info.pop("research_run_events", None)


def create_worker_sessionmaker() -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    """Create a pooled engine/session maker owned by one worker process loop."""
    worker_engine = create_async_engine(
        settings.DATABASE_URL,
        **database_engine_kwargs(role="worker"),
    )
    instrument_engine(role="worker", sync_engine=worker_engine.sync_engine)
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
