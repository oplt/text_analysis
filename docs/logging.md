# Backend logging

Centralized logging lives in `backend/core/logging.py` and is initialized via `setup_logging()` for the FastAPI app and Celery workers.

## Where logs go

| Destination | Controlled by | Default |
|-------------|---------------|---------|
| Console (stderr) | `LOG_TO_CONSOLE` | enabled |
| File | `LOG_TO_FILE` | enabled |
| File path | `LOG_FILE_PATH` | `logs/logs.txt` (relative to `backend/`) |

Example on disk: `backend/logs/logs.txt`.

## Environment variables

```env
LOG_LEVEL=INFO
LOG_TO_CONSOLE=true
LOG_TO_FILE=true
LOG_FILE_PATH=logs/logs.txt
LOG_RETENTION_DAYS=1
LOG_FORMAT=text
SLOW_REQUEST_MS=1000
SLOW_JOB_MS=5000
SLOW_EXTERNAL_CALL_MS=3000
```

- **`LOG_LEVEL`**: `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL`. Invalid values fall back to `INFO`.
- **`LOG_RETENTION_DAYS`**: Number of days of rotated log files to keep (1–365). Invalid values fall back to `1`.
- **`LOG_TO_FILE=false`**: Disables file output; console logging still works when `LOG_TO_CONSOLE=true`.

## Daily rotation

When file logging is enabled, logs rotate at **UTC midnight** into files named:

```text
logs/logs.txt.YYYY-MM-DD
```

On startup (and whenever logging is configured), rotated files older than the retention window are deleted. Only files matching `logs.txt.YYYY-MM-DD` in the log directory are removed.

## Request / correlation IDs

HTTP requests use `CorrelationIdMiddleware`:

1. Reuse `X-Correlation-ID` or `X-Request-ID` from the client when present.
2. Otherwise generate a UUID.
3. Attach the ID to `request.state.correlation_id` and a context variable used by log formatters.
4. Return the ID in both `X-Correlation-ID` and `X-Request-ID` response headers.

Log lines include `[correlation_id=…]`. Use the ID from the API response or log line to trace a request through services and workers.

## Slow operation warnings

| Threshold | Variable | Used for |
|-----------|----------|----------|
| HTTP requests | `SLOW_REQUEST_MS` | Request logging middleware |
| Celery jobs | `SLOW_JOB_MS` | Worker task completion |
| External / RAG calls | `SLOW_EXTERNAL_CALL_MS` | RAG retrieval and `timed_operation` helper |

## What not to log

Never log:

- passwords, tokens, API keys, or secrets,
- raw `Authorization` headers,
- full LLM prompts/responses or document bodies by default.

A redaction filter masks common secret patterns in log messages. Prefer logging IDs, counts, durations, and status—not payload content.

## Tests

```bash
cd backend
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. .venv/bin/python -m pytest tests/test_logging.py -q
```

## Key modules

| Module | Role |
|--------|------|
| `backend/core/logging.py` | Root logger setup |
| `backend/core/log_handlers.py` | File handler, rotation cleanup |
| `backend/core/log_context.py` | Correlation ID context |
| `backend/core/log_redaction.py` | Secret redaction |
| `backend/core/log_timing.py` | Latency helper |
| `backend/api/middleware/correlation_id.py` | Request ID middleware |
| `backend/api/middleware/request_logging.py` | Request access logs |
| `backend/workers/logging_hooks.py` | Celery job lifecycle logs |
