# ADR 0002: Celery With Redis For Asynchronous Jobs

## Status
Accepted

## Context
The backend already performs side-effect work that should not block request latency, especially transactional email delivery for verification and password reset flows. The system already depends on Redis for rate limiting and one-time token storage, so the architecture has an available distributed coordination component.

## Decision
Adopt Celery as the default background job executor and use Redis as both the broker and result backend.

Initial scope:

- run Celery workers as a separate runtime component
- route outbound email sending through Celery tasks
- keep request-time template rendering in the application service layer
- keep Redis as the single local-development infra dependency for both cache/token and queue concerns

Configuration is driven through:

- `CELERY_BROKER_URL`
- `CELERY_RESULT_BACKEND`
- `CELERY_TASK_ALWAYS_EAGER`
- `CELERY_TASK_DEFAULT_QUEUE`
- `CELERY_EMAIL_QUEUE`

If Celery eager mode is enabled, queued work runs inside the API process for lightweight local execution and test scenarios:

- Email tasks use `asyncio.create_task` when the API event loop is already running, otherwise a one-shot `asyncio.run` in the caller thread.
- Document indexing tasks spawn a daemon background thread that runs `asyncio.run` so upload/index HTTP handlers return immediately without blocking the request loop.

Set `CELERY_TASK_ALWAYS_EAGER=false` and run dedicated Celery workers in production.

## Consequences

Positive:

- Request latency is decoupled from SMTP/network behavior
- Email delivery can retry independently of the originating HTTP request
- The project now has a clear async execution boundary for future jobs such as webhooks, exports, or scheduled notifications
- Redis remains a shared and already-operational infrastructure dependency

Trade-offs:

- A worker process becomes part of the operational topology
- Monitoring and retry behavior now matter for correctness of side effects
- Redis is now more central to system health because it backs both app-level ephemeral state and async job dispatch
