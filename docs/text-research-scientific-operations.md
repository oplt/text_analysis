# Text Research — Scientific & Operational Guide

This document covers annotation campaigns, reliability statistics, leakage
rules, caching, run events, model lifecycle, drift, prediction layers, and
reproducibility guarantees for the Text Research module.

Companion overview: [`text-research.md`](text-research.md).

---

## Annotation Campaign workflow

Reliability and assignment are scoped to an **AnnotationCampaign**, not the
whole corpus. Mixing pilot rounds, different unit types, codebook versions, or
annotator pools into one agreement run is a study-design error the product
guards against by requiring campaign context.

### Concept

| Field | Role |
|-------|------|
| `codebook_id` / `codebook_version` | Frozen codebook identity for the wave |
| `unit_type` | e.g. `paragraph` / `sentence` — assignments and reliability stay on this grain |
| `annotator_ids` | Membership for the campaign |
| `assignment_strategy` | `overlap` / `shared` / `disjoint` (plus sampling knobs) |
| `annotation_mode` | `blind_reliability` or `ai_assisted` |
| `status` | `draft` → `active` → `completed` / `archived` |

### Lifecycle

1. **Create** — `AnnotationCampaignService.create_campaign` starts `active` with
   `started_at` set. If a codebook is attached, `codebook_version` is snapshotted
   from the codebook at create time.
2. **Assign** — `assign` plans tasks for campaign members (or an explicit
   annotator list), always using the campaign `unit_type` and
   `assignment_strategy`. Tasks store `campaign_id`.
3. **Progress** — `progress` returns per-annotator / status counts and
   `completion_rate = completed_tasks / total_tasks`.
4. **Complete** — `patch_campaign(status=completed)` sets `completed_at`.

API surface lives under `/api/v1/research` campaign routes; the Annotation UI
setup tab drives create/assign.

---

## Blind vs AI-assisted annotation

Modes are **mutually exclusive**.

| Mode | `blind_mode` | `ai_assistance_enabled` |
|------|--------------|-------------------------|
| `blind_reliability` (default) | true | false |
| `ai_assisted` | false | true |

### What blind coding hides

For an incomplete task on a blind campaign, the policy exposes:

- `hide_peer_annotations` — other coders’ labels
- `hide_adjudications` — gold / adjudicated labels
- `hide_model_predictions` — model suggestions

Model predictions stay hidden for the **entire** blind campaign (even after the
annotator completes their task). Peer annotations and adjudications may become
visible after the annotator’s own task is `completed` (post-coding review).

Backend enforcement:

- Annotation list filters peers when `hide_peer_annotations` is set.
- Active-learning / uncertain queue returns `[]` when the campaign is blind or
  the unit policy hides predictions.
- Frontend `shouldFetchPredictions` / `shouldFetchPeerAnnotations` /
  `shouldFetchAdjudications` refuse queries under the same flags.

Do not treat blind mode as a UI-only toggle — the API must not leak the hidden
layers.

---

## Reliability statistic selection

**Measurement level today: nominal only.** Ordinal / interval / weighted kappa
are not implemented. Summary metadata records `scale: "nominal"`.

### Which statistic when

| Design | Primary chance-corrected index | Also reported |
|--------|--------------------------------|---------------|
| Exactly **two** coders with overlapping units | **Cohen's κ** | Raw agreement; **Krippendorff's α** |
| **Three or more** coders (fixed-*n* nominal design) | **Fleiss' κ** | Pairwise Cohen where useful; **Krippendorff's α** |
| Missing / ragged coder×unit cells | Prefer **Krippendorff's α** | Fleiss may exclude non-modal rows |

Plain-language rules:

- **Cohen's kappa** — exactly two coders on the same units.
- **Fleiss' kappa** — three or more coders under a fixed-*n* nominal design
  (units that do not match the modal rater count are excluded and warned).
- **Krippendorff's alpha** — two or more coders; supports missing and ragged
  annotations via the coincidence matrix.

Campaign / unit-type / codebook-version filters are applied in SQL when loading
annotation rows so agreement cannot accidentally mix study waves.

### Scientific warnings

Reliability results emit human-readable `Warning:` diagnostics for small
overlap, high missingness, single-category / dominant-class instability, wide
CIs, and Fleiss design violations. These are interpretation aids, not soft
substitutes for preventing invalid designs.

---

## Confidence intervals

Agreement indices that support resampling attach a **percentile bootstrap CI**:

- Default `bootstrap_samples=2000`, `confidence_level=0.95`
- Default `random_seed=42` when unset (stored on the `AnalysisRun`)
- Seed offsets keep raw / κ / Fleiss / α replicates independent (`seed`,
  `seed+1`, `seed+2`, `seed+3`)

Each CI payload includes `lower`, `upper`, `confidence_level`,
`bootstrap_samples`, `random_seed`, and `n_replicates`. Run metrics summarize
the bootstrap settings for provenance.

---

## Feature-selection leakage rules

**Invariant — never violate:**

```text
vectorizer.fit          → training partition only
feature_selector.fit    → training partition only
threshold tuning        → validation / designated tuning fold only
hyperparameter tuning   → nested / inner fold only
```

Held-out validation and test partitions may only `.transform()`.

Nested grouped CV refits the selector independently on each outer-fold training
set. Leakage is **hard-prevented and tested**; the product does **not** emit a
soft “feature selection fitted outside training fold” warning as if the leak
were allowed. Classifier scientific warnings cover vocabulary / sparsity /
prevalence diagnostics instead.

Stage cache refuses supervised / leakage-sensitive stage names so fitted
vectorizers or selectors cannot be shared across folds via cache identity.

---

## Caching architecture

Content-addressable **stage cache** (`infrastructure/stage_cache.py`):

| Layer | Store |
|-------|--------|
| L1 | In-process LRU |
| L2 | Redis meta / status / distributed locks |
| L3 | Shared artifact directory or object storage |

Cache identity hash includes at least:

- `engine_version`
- `stage_name`
- `input_checksum` (corpus / snapshot identity)
- `spec_hash`
- preprocessing / params

Same identity → hit. Changed preprocessing, corpus snapshot, or engine version
→ miss. `get_or_compute` + Redis locks prevent stampede duplicate work.

---

## Redis event architecture

PostgreSQL remains the durable source of truth for analysis runs. Redis carries
ephemeral progress notifications.

- Channel: `research:run:{run_id}:events`
- Envelope: `{ "event": "<name>", "run": {…}, "published_at": "…" }`
- Named events: `queued`, `started`, `progress`, `artifact-created`,
  `completed`, `failed`, `cancelled`

SSE endpoint `GET /api/v1/research/runs/{run_id}/events`:

1. Prefer Redis Pub/Sub
2. Periodically **reconcile from DB** (≈15s) so missed publishes still surface
3. If Redis is unavailable, fall back to DB reconcile with a short sleep
4. Emit SSE keepalive comments so proxies do not idle-close the stream

Frontend: EventSource via `useRunEvents`; when SSE is unavailable,
`activeRunRefetchInterval` polls every 2s for non-terminal runs.

---

## Worker concurrency policy

Long research jobs run on Celery. Scale throughput with worker **process**
concurrency (`CELERY_CONCURRENCY` / `--concurrency`). Inside each process keep
BLAS / OpenMP / sklearn / joblib thread counts at **1** so processes do not
oversubscribe CPU (`backend/workers/parallelism.py`, applied on worker init).

Relevant settings:

- `RESEARCH_APPLY_THREAD_LIMITS`
- `RESEARCH_WORKER_BLAS_THREADS`
- `RESEARCH_SKLEARN_N_JOBS`
- `RESEARCH_JOBLIB_N_JOBS`

`execution_policy.py` covers retry / timeout / checkpoint / idempotency by
resource class — not process concurrency.

---

## Model lifecycle

Trained classifiers move through:

```text
candidate → approved → deprecated
```

- Invalid statuses are rejected (`422`).
- Approving with `deprecate_others=True` deprecates sibling models in the same
  project/corpus/`task_type` with the same label-set hash.
- Lifecycle notes and `lifecycle_updated_at` are persisted for audit.

UI: `/research/:projectId/models` (Model Registry).

---

## Drift interpretation

Drift monitoring compares baseline vs current prediction / score / feature
summaries (TVD, PSI-like, KS or mean–std fallback, Jaccard on terms).

**Distribution drift is a review signal, not proven model degradation.** Do not
auto-retire models solely on distribution shift; confirm with labeled
performance or human review. Reports persist as `drift_monitoring` analysis
runs. UI: `/research/:projectId/drift`.

---

## Prediction provenance

Predictions are stored separately from human annotations and adjudications.

| Layer | Store | Never merged into |
|-------|--------|-------------------|
| MODEL | `ModelPrediction` / `PredictionSet` | Annotation / Adjudication tables |
| HUMAN | `Annotation` | Prediction tables |
| GOLD | `Adjudication` | Prediction or Annotation overwrite |

A **PredictionSet** is a header per prediction analysis run linking
`trained_model_id`, `model_version`, `dataset_snapshot_id`, and
`analysis_run_id`. Listing helpers keep layers filterable side-by-side in the
Prediction Sets UI (`/research/:projectId/predictions`).

---

## Reproducibility guarantees

Runs and exports attach provenance schema **v2**, including:

- `git_commit`, `application_version`, `python_version`
- `package_lock_checksum` (prefer `backend/uv.lock` SHA-256; CI uses
  `uv sync --frozen`)
- optional container / Docker image digest
- `engine_version`, corpus snapshot id/hash
- campaign / codebook / unit type / filters
- preprocessing config **hash**, feature and selection configs
- algorithm, hyperparameters, validation strategy, split hashes
- artifact checksums and analysis `random_seeds`

Reliability and classification persist the analysis `random_seed` on
`AnalysisRun`. Re-running with the same seed, lockfile, engine version, and
corpus snapshot is the supported reproducibility path.

Performance regression measurements (not hard SLOs): see
[`text-research.md`](text-research.md) benchmarks (`make bench-workflows`,
`make bench-1k`).

---

## Related code

| Topic | Primary modules |
|-------|-----------------|
| Campaigns / blind policy | `application/campaign_service.py` |
| Reliability | `application/reliability_service.py`, `infrastructure/reliability.py` |
| Scientific warnings | `infrastructure/scientific_warnings.py` |
| Leakage / selection | `infrastructure/classifiers.py`, `tests/test_feature_selection.py` |
| Stage cache | `infrastructure/stage_cache.py` |
| Run events | `infrastructure/run_events.py`, `api/routes.py` (`stream_run_events`) |
| Workers | `backend/workers/parallelism.py` |
| Lifecycle | `application/model_lifecycle_service.py` |
| Drift | `infrastructure/drift_monitoring.py`, `application/drift_service.py` |
| Prediction sets | `application/prediction_set_service.py` |
| Provenance | `infrastructure/provenance.py` |
