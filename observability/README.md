# Local Observability

This repository runs observability locally on the host machine. It does not use
Docker for Prometheus, Grafana, or Tempo.

## Services

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3001
- Tempo API: http://localhost:3200
- Tempo OTLP HTTP: http://localhost:4318
- Tempo OTLP gRPC: localhost:4317
- Observability Hub: http://localhost:5173/observability

## Required Binaries

Install these tools locally and make sure they are available on `PATH`:

```bash
prometheus --version
grafana-server -v
tempo --version
```

Install notes:

- Prometheus: download the release archive for your OS from the Prometheus
  releases page, then put the `prometheus` binary on `PATH`.
- Grafana: install Grafana OSS using your OS package manager, Homebrew, or the
  Grafana release package. This setup uses the `grafana-server` command.
- Tempo: download the Tempo release for your OS from Grafana Labs releases, then
  put the `tempo` binary on `PATH`.

You can override binary paths when starting local development:

```bash
make local-dev PROMETHEUS_BIN=/path/to/prometheus TEMPO_BIN=/path/to/tempo GRAFANA_BIN=/path/to/grafana-server
```

## Start Everything

```bash
make local-dev
```

This starts the existing backend, frontend, Redis, and Celery processes, plus
Prometheus, Grafana, and Tempo through the existing honcho startup flow.

Grafana local credentials are set to:

```text
admin / admin
```

## Observability-Only Commands

```bash
make observability-up
make observability-down
make observability-logs
```

`observability-up` starts only Prometheus, Grafana, and Tempo.

## Verify Metrics

Generate traffic against the backend, then check the FastAPI metrics endpoint:

```bash
curl http://localhost:8000/metrics
```

Open Prometheus:

```text
http://localhost:9090
```

Check targets:

```text
http://localhost:9090/targets
```

The `fastapi-backend` target should be `UP`.

## Verify Traces

1. Call a few backend endpoints, such as `http://localhost:8000/health`.
2. Open Grafana at `http://localhost:3001`.
3. Go to Explore.
4. Select the Tempo datasource.
5. Search for service name `fastapi-backend`.

## Backend OpenTelemetry Defaults

The local `Procfile.dev` passes these values to the backend:

```env
OTEL_SERVICE_NAME=fastapi-backend
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
OTEL_TRACES_EXPORTER=otlp
```

The older `OTLP_ENDPOINT` setting is still supported for gRPC exporters.

## Observability Hub

The authenticated app includes an Observability Hub at:

```text
http://localhost:5173/observability
```

The Hub is an operational summary and launchpad. It shows lightweight health
cards for the API, database, Redis, workers, metrics, latency, and local
observability tools. Grafana is the primary investigation UI. Prometheus is
shown only to admins as a raw PromQL debug tool. Tempo trace investigation opens
through Grafana Explore.

Grafana is opened in a new tab instead of embedded in an iframe. This keeps the
first implementation simpler and avoids extra cookie, SSO, cross-origin,
clickjacking, and Grafana RBAC configuration.

The frontend does not hardcode observability URLs. It loads them from:

```text
GET /api/v1/observability/links
GET /api/v1/observability/status
```

## Hub Environment Variables

```env
GRAFANA_PUBLIC_URL=http://localhost:3001
PROMETHEUS_PUBLIC_URL=http://localhost:9090
TEMPO_PUBLIC_URL=http://localhost:3200

GRAFANA_APP_OVERVIEW_DASHBOARD_PATH=/d/fastapi-overview/fastapi-overview
GRAFANA_API_DASHBOARD_PATH=/d/fastapi-overview/fastapi-overview
GRAFANA_FRONTEND_DASHBOARD_PATH=
GRAFANA_DATABASE_DASHBOARD_PATH=
GRAFANA_CACHE_DASHBOARD_PATH=
GRAFANA_WORKERS_DASHBOARD_PATH=
GRAFANA_SCHEDULED_TASKS_DASHBOARD_PATH=
GRAFANA_ERRORS_DASHBOARD_PATH=/d/fastapi-overview/fastapi-overview
GRAFANA_TEMPO_EXPLORE_PATH=/explore
```

Empty dashboard paths are returned as `Not configured` in the Hub.

## Grafana Dashboard Variables

The Hub adds safe dashboard query parameters when they are available:

```text
orgId=1
from=now-1h
to=now
var-service=<service>
var-environment=<environment>
var-route=<route>
var-job=<jobName>
```

Trace and request IDs are kept for trace exploration links, not Prometheus
dashboard variables.

## Access Expectations

- Normal authenticated users can open the Hub and see high-level status cards.
- Admin users can open Grafana dashboards, Tempo Explore, and Prometheus Debug.
- Missing or unauthorized tools appear as disabled or hidden actions.
- No credentials, cookies, or access tokens are placed in external URLs.

## Troubleshooting Missing Configuration

If a shortcut says `Not configured`, set the matching `GRAFANA_*_DASHBOARD_PATH`
or public tool URL in `backend/.env`, then restart `make local-dev`.
