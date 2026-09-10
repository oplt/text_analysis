# Text Research Correctness Audit

Date: 2026-09-10

## Scope and result

The audit traced the Text Research training path from frozen dataset snapshot
through grouped splitting, worker execution, persisted runs, artifacts, and
provenance. It found and repaired one split-planning failure: a small,
class-balanced corpus could be marked stratification-feasible under the
two-group minimum, while a default 20% holdout requested five stratified folds.
`StratifiedGroupKFold` then raised before a fallback could be selected.

`plan_grouped_splits` and nested grouped CV now evaluate feasibility against
the actual requested fold count. When that count cannot be supported, they use
the existing group-safe fallback and record its reason. This preserves the
no-document-leakage contract and avoids a runtime failure.

## Verified invariants

- Dataset construction freezes labels, unit IDs, campaign snapshot identity,
  and content hashes before model training; it does not train from mutable
  current annotations.
- All classifier train/validation/test partitions assert disjoint document
  groups. Multilabel targets use `GroupShuffleSplit`, never standard
  stratification, and record that decision in split provenance.
- Training and quantitative workers reload run state at cooperative
  checkpoints. A user cancellation persists `cancelled`, and the worker
  preserves that terminal state rather than reporting a failed run.
- Run records carry specification, seed, snapshot, environment, artifact, and
  partition provenance. Artifacts are persisted and tested independently of
  the in-process cache.

## Validation

- Split, classifier, nested-CV, cancellation, run-event, queue, and workflow
  tests: 82 passed.
- Dataset snapshot, provenance, artifact-store, stage-cache, and execution
  lifecycle tests: 43 passed.

End-to-end services and optional R/quanteda execution are outside this audit.
