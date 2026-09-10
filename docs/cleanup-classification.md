# Phase 2 Cleanup Classification

Date: 2026-09-10

This classification is based on source callers, route/worker registration, and
targeted tests. It deliberately does not remove behavior merely because a
static search found `pass`, `NotImplementedError`, or an old-looking document.

## KEEP

- `BaseAiProvider` abstract methods: concrete local, OpenAI, and Anthropic
  providers are instantiated by `AiProviderRegistry` and selected by AI run
  services.
- Optional observability `try`/`except` blocks in research SSE and stage-cache
  code: they keep analysis execution available when metrics integrations are
  absent.
- Provenance environment probes: absent package, image, or version metadata is
  intentionally represented as unavailable rather than blocking a run.
- Grouped-split, cancellation, artifact, and plugin-registry helpers: each has
  active application/worker callers and targeted tests.

## RETIRED FINDINGS

- `docs/performance-audit.md` was corrected in Phase 4 with measured 1k, 10k,
  and 100k DFM results, current test coverage, and remaining limits.
- `DESIGN.md` was corrected in Phase 6 to describe the implemented Text
  Analysis application shell and MUI visual system. The old Tesla marketing
  page description was not an application contract.

## PHASE 6 DECISIONS

- The legacy `/api/v1/ai/documents*` routes and `LegacyAiDocumentService` are
  not removable: registered endpoints and `AiDocumentService` callers retain a
  public compatibility contract while routing exclusively through RAG storage,
  ingestion, and retrieval. The older `AiDocument`/`AiDocumentChunk` metadata
  mappings likewise remain pending an explicitly planned schema migration.
- No source-verified duplicated helper was safe to merge. In particular, the
  text-research pipeline and execution-policy compatibility exports preserve
  existing import contracts and have regression coverage. No dependency was
  identified as unused by a confirmed runtime path.

## REMOVE / COMPLETE

No source-verified runtime candidate is safe to remove or requires a stub
completion in this phase. The only confirmed code defect was the stratified
fold-count failure recorded in `correctness-audit.md` and repaired in the split
planner.

## Final report status

Phase 7 final validation did not discover a safe cleanup candidate. It did
identify a blocked asynchronous stage-cache test and an ambiguous E2E locator;
those are correctness/test-maintenance follow-ups, not evidence that the
classified compatibility paths or dependencies can be removed.
