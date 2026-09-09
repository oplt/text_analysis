# Text-research architecture gap review (implementation status)

Source: user gap analysis in `prompt.txt` (2026-09-09).

## P0 — Critical / High — Done

| TODO | Status |
|------|--------|
| AnalysisSpecification v2 + hash + PipelineCompiler | Done |
| CorpusSnapshot / DatasetView / PreparedCorpusArtifact | Done |
| Topic/classification consume prepared artifacts | Done |
| Quantitative analyses consume prepared artifacts | Done |
| StageRunner executes compiled plans | Done |
| analysis_spec_hash + checksums on runs | Done |
| Typed artifact registry | Done |
| Split feasibility + stratified-group logic | Done |
| Generic topic `group_by` | Done |
| Pipeline/property + golden tests | Done |

## P1 — Analytical expansion — Done

| TODO | Status |
|------|--------|
| Full TextTransform pipeline (string + token stages) | Done |
| Document + unit-level language detection | Done |
| Optional spaCy engine + `[nlp]` extra | Done |
| Embedding providers + embedding LR/SVM API | Done |
| Optional BERTopic + decomposed `semantic_stack` | Done |
| Embedding cache | Done |
| Topic transform + holdout perplexity API | Done |
| Hungarian + cosine/doc-distribution stability | Done |
| Nested grouped CV as train `validation_strategy` | Done |
| Metadata-sliced error analysis | Done |
| Threshold objectives incl. `custom_utility` + abstention | Done |
| Optional transformer classifier (honest failure) | Done |

## P2 — Platform / scalability — Done

| TODO | Status |
|------|--------|
| Stage cache, out-of-core, Parquet/Arrow | Done |
| Dedicated research_* Celery queues | Done |
| Nested parallelism controls (BLAS/sklearn/joblib) | Done |
| Strengthened provenance + one-click reproduce | Done |
| Retry/checkpoint/idempotency | Done |
| Plugin registry + workflow recipes | Done |
| Model lifecycle states | Done |
| Drift monitoring API | Done |
| Headless CLI (`compile-spec`, `run-analysis`, …) | Done |

## Annotation loop — Done (core)

| TODO | Status |
|------|--------|
| PredictionSet first-class table + API | Done |
| Predictions never overwrite gold annotations | Done |

## Testing / CI — Done

| TODO | Status |
|------|--------|
| Golden multilingual pipeline tests | Done |
| Property tests (leakage, checksums, immutability) | Done |
| Quanteda Python fixtures | Done |
| Optional R quanteda CI workflow | Done (skips when R absent) |
| Current-HEAD quality gates (`.github/workflows/ci.yml`) | Done |
| Scale benchmarks (1k / 10k / optional 100k) | Done |

## Remaining intentional limitations (not incomplete tasks)

These are **optional heavy deps / future polish**, not unfinished roadmap items:

- Real UMAP/HDBSCAN only when `umap-learn` / `hdbscan` installed (SVD/KMeans fallback otherwise)
- Real BERTopic / transformers only with optional extras; cores stay sklearn-native
- Full Argilla/Label-Studio UX rewrite is out of scope (backend PredictionSet covers the data contract)
- Default CI does not require R; R parity is opt-in via `.github/workflows/quanteda-parity.yml`
- 100k benchmark is opt-in (`BENCHMARK_INCLUDE_100K=1` or workflow_dispatch)

## Tests

```bash
# Unit suite (current HEAD)
make test-backend

# Frontend
make test-frontend

# Lint + typecheck
make check

# Scale benches (1k always; 10k via BENCHMARK_INCLUDE_10K=1)
make bench-1k
make bench-10k

# GitHub Actions: .github/workflows/ci.yml
# Always checks out and reports the triggering commit SHA — do not treat an
# older green run as proof that current HEAD is healthy.
```

Latest: **482 passed, 2 skipped** (R parity skips) + benchmark suite.

## Ops

```bash
alembic upgrade head   # includes lifecycle + prediction_sets
# workers:
#   -Q research_light,research_cpu,research_memory,research_gpu
PYTHONPATH=. python -m backend.modules.text_research.cli --help
```
