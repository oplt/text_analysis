# Text Research Performance and Scalability Audit

Date: 2026-09-10

## Measurements

Measurements used the deterministic Text Research benchmark suite on Linux
(x86_64, Python 3.12.3, 16 logical CPUs). They exercise preprocessing, the
hashing DFM path, stage-cache reuse, reliability, classification preparation,
and prediction-row serialization. They do not include a live PostgreSQL,
Redis, Celery, browser, or R runtime.

| Workflow | Scale | Result |
| --- | ---: | --- |
| Prepare + hashing DFM | 1,000 units | 0.038 s total, cache reuse observed |
| Prepare + hashing DFM | 10,000 units | 0.484 s total, cache reuse observed |
| Prepare + hashing DFM | 100,000 units, before | 3.898 s total, 262.9 MiB process RSS growth |
| Prepare + hashing DFM | 100,000 units, after | 3.882 s total, 262.2 MiB process RSS growth |

The 100k wall-clock and peak-RSS difference is within normal process and host
noise. The completed optimization instead has a deterministic work reduction:
the hashing DFM path no longer creates a second full-corpus shallow copy before
it starts processing batches.

## Completed improvements

- `build_hashing_matrix` now passes the caller's `Sequence` directly to the
  existing batch iterator and passes each bounded batch directly to the
  vectorizer. It preserves sparse output and removes an O(n) pointer copy.
- Repository batch helpers no longer copy whole ID lists before slicing. The
  existing 500-item SQL `IN` batches are retained.
- `corpus_ids_for_text_units` now uses those bounded batches rather than one
  unbounded `IN` predicate, protecting large annotation assignments from bind
  parameter limits.

## Source audit

| Area | Finding | Status |
| --- | --- | --- |
| Cache and async work | The async stage-cache path moves blocking L3/Redis I/O and synchronous factories to worker threads, with async lock waiting. | Verified |
| Database pools | API and worker pools, overflow, recycle, and timeout settings are explicit; worker guidance accounts for Celery process count. | Verified |
| Worker parallelism | Resource-class queues and BLAS thread limits prevent default nested parallelism. | Verified |
| Large-corpus DFM | Hashing vectorization stays CSR and avoids vocabulary materialization; dense exports are capped. | Improved |
| Frontend loading | Routes and heavy analysis panels are lazy-loaded. | Verified |

## Remaining measured constraints

### P1: prediction still materializes the candidate corpus

`PredictionService.execute_prediction` loads all documents and text units before
its later transform/upsert batching. This keeps prediction-set membership and
its frozen result contract simple, but peak memory still grows with the selected
corpus. A safe fix must introduce a persisted candidate snapshot or a paged
prediction-set builder, then stream repository pages through model transforms.
It was not changed here because changing that membership contract without an
end-to-end PostgreSQL workflow test would risk reproducibility.

### P2: list-returning corpus APIs remain materializing adapters

`list_text_units_for_corpus` pages its database reads for large corpora, but
returns an aggregate list for legacy callers. New high-volume callers should
consume `iter_text_units_for_corpus`; converting existing quantitative and
prediction workflows requires the same snapshot-aware streaming design above.

### P3: production browser and service metrics are unavailable locally

No running isolated API/PostgreSQL/Redis/browser stack was available, so this
audit does not claim Core Web Vitals, query latency, queue depth, or real
database query counts. Existing benchmark rows correctly report zero database
queries because they are pure in-process measurements.

## Validation

- Benchmark suite: 10 passed, 2 opt-in large-scale tests skipped.
- 100k DFM benchmark before and after: passed.
- Stage-cache, out-of-core, large-corpus repository, and benchmark tests after
  the change: 37 passed, 2 skipped.
- Phase 7 reran the repository 1k and 10k benchmark gates successfully.
- Phase 7 could not complete the wider backend or Text Research suite because
  `AsyncStageCacheTests.test_get_or_compute_async_materializes_payload`
  exceeded the configured 30-second faulthandler threshold. This does not
  invalidate the benchmark measurements above, but prevents claiming a final
  full-suite pass.
