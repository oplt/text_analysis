# RAG scale envelope

The reproducible load harness is `backend.modules.rag.eval.load_benchmark`.
It defines 10k, 100k, and 1M chunk scales and streams deterministic synthetic rows so corpus
generation itself does not dominate memory measurements.

## Methodology

1. Choose a `LoadScale` from `DEFAULT_SCALES` (10k / 100k / 1M chunks).
2. Stream rows via `synthetic_chunk_rows` — never materialize the full corpus in one list.
3. Callers supply real DB insert / embed / retrieve operations to `measure_operation`.
4. Record indexing/embedding throughput, retrieval P50/P95/P99, PostgreSQL CPU/IO,
   scope-resolution memory/time, and synthesis scheduling time.
5. Use `memory_ceiling_smoke` for CI-safe assertions on a bounded sample prefix.

## Provisional envelope (NEEDS_LIVE_MEASUREMENT)

| Scale | Docs | Chunks | Indexing throughput | Retrieval P95 | Scope memory | Status |
|-------|------|--------|---------------------|---------------|--------------|--------|
| 10k | 100 | 10_000 | NEEDS_LIVE_MEASUREMENT | NEEDS_LIVE_MEASUREMENT | NEEDS_LIVE_MEASUREMENT | provisional |
| 100k | 1_000 | 100_000 | NEEDS_LIVE_MEASUREMENT | NEEDS_LIVE_MEASUREMENT | NEEDS_LIVE_MEASUREMENT | provisional |
| 1M | 10_000 | 1_000_000 | NEEDS_LIVE_MEASUREMENT | NEEDS_LIVE_MEASUREMENT | NEEDS_LIVE_MEASUREMENT | provisional |

Before production limits are changed, run each scale against the pgvector-enabled integration
database and replace the NEEDS_LIVE_MEASUREMENT placeholders. Results must include environment,
database/index settings, and query profile. No threshold in production is justified until those
measurements are recorded.
