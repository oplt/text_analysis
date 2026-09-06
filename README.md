# Generic App

Generic full-stack starter with:

- FastAPI backend
- React + Vite frontend
- PostgreSQL, Redis, and MinIO for local infrastructure
- Celery workers for asynchronous jobs, using Redis as broker/result backend
- JWT auth with refresh rotation
- Admin settings, notifications, profile, and project modules
- Optional platform modules for billing, API keys, webhooks, feature flags, and email templates
- Sentry/OpenTelemetry hooks and S3-compatible avatar storage

## Local Setup

1. Start infrastructure:

```bash
cp infra/.env.example infra/.env
docker compose -f infra/docker-compose.yml up -d
```

Mailpit is included for local email capture:

- SMTP server: `localhost:1025`
- Web inbox: `http://localhost:8025`

2. Configure the backend:

```bash
cd backend
cp .env.example .env
uv sync
.venv/bin/alembic upgrade head
```

See [docs/logging.md](../docs/logging.md) for log file location, rotation, and correlation IDs.

3. Start the backend:

```bash
cd backend
.venv/bin/uvicorn backend.api.main:app --reload
```

4. Start the Celery worker:

```bash
cd backend
.venv/bin/celery -A backend.workers.celery_app:celery_app worker --loglevel=INFO --queues=default,email
```

5. Configure the frontend:

```bash
cd frontend
cp .env.example .env
npm install
```

6. Start the frontend:

```bash
cd frontend
npm run dev
```

Or start the full local development stack, including Prometheus, Grafana, and Tempo:

```bash
make local-dev
```

Observability setup and verification steps are documented in
[observability/README.md](observability/README.md).

## Notes

- Local object storage uses MinIO on `http://localhost:9000` and its console on `http://localhost:9001`.
- Local infrastructure secrets now come from `infra/.env`; the compose file no longer embeds credentials.
- Redis now serves both app-level caching/token storage and the Celery broker/result backend.
- The first Celery-backed workflow is outbound email delivery for verification and password reset flows.
- Local `.env.example` defaults to Mailpit plus `CELERY_TASK_ALWAYS_EAGER=true`, so signup/reset emails work without a separate worker.
- Avatar uploads are stored in the configured S3-compatible bucket instead of a placeholder path.
- `/admin/platform` lets you rename the app, rename the core domain labels, pick a module pack, and manage plans, flags, and email templates.
- Set `ADMIN_SIGNUP_INVITE_CODE` in `backend/.env` to allow invite-only admin registration during sign-up.
- Authentication now uses `httpOnly` cookies plus a CSRF token cookie/header pair for state-changing requests.
- Module packs are intended for clone-time reuse:
  - `lean_saas`
  - `automation_suite`
  - `client_portal`
  - `full_platform`
- Observability is enabled through backend config:
  - `SENTRY_DSN`
  - `SENTRY_TRACES_SAMPLE_RATE`
  - `OTLP_ENDPOINT`
  - `OTLP_INSECURE`
  - `OTEL_SERVICE_NAME`
  - `OTEL_EXPORTER_OTLP_ENDPOINT`
  - `OTEL_EXPORTER_OTLP_PROTOCOL`
  - `OTEL_TRACES_EXPORTER`
  - `GRAFANA_PUBLIC_URL`
  - `PROMETHEUS_PUBLIC_URL`
  - `TEMPO_PUBLIC_URL`
