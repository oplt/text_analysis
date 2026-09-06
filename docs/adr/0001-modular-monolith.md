# ADR 0001: Modular Monolith With Bounded Contexts

## Status
Accepted

## Context
The project needs clear domain boundaries, predictable local development, and a path to production observability without the operational cost of early microservices. The current backend already trends in that direction with module-level routers, services, repositories, and schemas.

## Decision
Adopt a pragmatic modular monolith as the default architecture.

Backend modules are the main unit of composition:

- `identity_access`
- `users`
- `profile`
- `projects` as the current core business domain
- `notifications`
- `audit`
- `admin`

Each module owns its router, service layer, repository layer, and DTO schemas. Shared concerns stay in `app/core`, `app/api/deps`, and `app/db`.

Authentication follows the current split:

- short-lived access token returned to the SPA
- rotating refresh token stored in an `HttpOnly` cookie
- refresh session persistence in PostgreSQL
- Redis for rate limiting and one-time verification/reset tokens
- Celery for asynchronous jobs, with Redis as broker/result backend and email delivery as the first queued workload

Observability is standardised around OpenTelemetry and Sentry hooks, enabled by configuration.

## Consequences

Positive:

- Faster delivery than microservices while preserving domain boundaries
- Easier local setup, migrations, and cross-module transactions
- Clear extraction seams if a module must later become a separate service
- Shared observability and security controls stay centralized

Trade-offs:

- Module boundaries must be enforced by convention and review
- A single deployable means careless coupling can still spread quickly
- Some frontend structure is still transitional and should move toward stronger feature colocation over time
