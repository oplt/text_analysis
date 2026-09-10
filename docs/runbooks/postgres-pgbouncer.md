# Postgres + PgBouncer for API and Celery workers

PostgreSQL remains the durable store. Under load, put **PgBouncer** between
application processes and Postgres so connection counts stay bounded.

## Why

Each FastAPI process and each Celery prefork child opens its own SQLAlchemy
async pool:

```text
API max ≈ DB_POOL_SIZE + DB_MAX_OVERFLOW
Worker max ≈ concurrency × (DB_WORKER_POOL_SIZE + DB_WORKER_MAX_OVERFLOW)
```

Without a pooler, scaling Celery `--concurrency` or API replicas exhausts
`max_connections` on Postgres.

## Recommended topology

```text
FastAPI  ──┐
           ├──► PgBouncer (transaction pooling) ──► PostgreSQL
Celery   ──┘
```

Point `DATABASE_URL` at PgBouncer (same DB name/user), not directly at
Postgres, in staging/production.

## Pool mode

Use **transaction** pooling for this codebase:

- Sessions are short-lived request/task scopes (async SQLAlchemy).
- Avoid session pooling features that pin server connections across requests
  (prepared statements that outlive a transaction, `LISTEN`, advisory locks
  held across commits).

If you must use those features, route them through a direct Postgres URL or
a separate session-mode pool.

## Suggested starting sizes

| Knob | Dev default | Production starting point |
|------|-------------|---------------------------|
| `DB_POOL_SIZE` | 5 | 5–10 per API process |
| `DB_MAX_OVERFLOW` | 10 | 5–10 |
| `DB_POOL_TIMEOUT` | 30 | 30 |
| `DB_POOL_RECYCLE` | 1800 | ≤ PgBouncer/`server_idle_timeout` |
| `DB_WORKER_POOL_SIZE` | 2 | 1–2 per Celery child |
| `DB_WORKER_MAX_OVERFLOW` | 2 | 0–2 |

Example: 2 API replicas + Celery `--concurrency=8`:

```text
API:    2 × (5 + 10) = 30
Worker: 8 × (2 + 2)  = 32
Client total toward PgBouncer ≈ 62
```

Size PgBouncer `default_pool_size` / `max_client_conn` and Postgres
`max_connections` from that math, leaving headroom for migrations, admins,
and replicas.

## Nested CPU parallelism (keep)

Do **not** “fix” DB pressure by raising sklearn/`n_jobs=-1` inside workers.
Keep:

```text
RESEARCH_WORKER_BLAS_THREADS=1
RESEARCH_SKLEARN_N_JOBS=1
RESEARCH_JOBLIB_N_JOBS=1
```

Scale CPU with Celery processes/queues; scale DB with PgBouncer + modest
SQLAlchemy pools.

## Connection budget (do not raise pools blindly)

```text
(API replicas × processes_per_replica × (DB_POOL_SIZE + DB_MAX_OVERFLOW))
+
(Celery worker deployments × --concurrency × (DB_WORKER_POOL_SIZE + DB_WORKER_MAX_OVERFLOW))
+
administrative / migration / replica connections
<
PostgreSQL max_connections   (or PgBouncer default_pool_size toward Postgres)
```

Prefer **more Celery processes with small worker pools** over large
`DB_POOL_SIZE`. Raising SQLAlchemy pools without PgBouncer headroom causes
checkout timeouts and saturation, not higher throughput.

## Observability (Prometheus)

Exposed when `PROMETHEUS_METRICS_ENABLED=true` (default):

| Metric | Meaning |
|--------|---------|
| `database_pool_checked_out_connections{role}` | Live checkouts (`api` / `worker`) |
| `database_pool_saturation_ratio{role}` | checkouts / (pool_size + max_overflow) |
| `database_pool_wait_seconds{role}` | Time spent waiting for a free connection |
| `database_pool_timeouts_total{role}` | Checkout timeouts |
| `database_pool_connection_failures_total{role}` | Acquire / invalidate failures |
| `database_query_duration_seconds{role}` | Cursor execute latency |
| `database_session_duration_seconds{role}` | Request/task session lifetime |

Alert when saturation stays high **and** wait/timeouts rise — that means
capacity or query duration problems, not “need a bigger pool” by default.

## Inspect live policy

```python
from backend.db.session import describe_database_pool_policy
from backend.workers.parallelism import describe_parallelism_policy

print(describe_database_pool_policy())
print(describe_parallelism_policy()["concurrency_policy"])
```
