# R/quanteda architecture status

## Current status: not implemented as an execution engine

The application has Python-native quantitative processing, a researcher-facing
quanteda script export, and optional parity fixtures. It does not have an R
worker, `Rscript` runner, R-specific job lifecycle, or an executable
backend/frontend R-analysis selection.

## Existing integration boundary

```text
Corpus + text units
  -> /api/v1/research/corpora/{corpus_id}/export/units.csv
  -> researcher-controlled CSV + generated quanteda script
  -> external R / quanteda session
```

`ExportService.build_quanteda_script` generates an independent replication
script and the `/export/quanteda-script` route returns it. The service is
explicit that the backend does not execute this script. Optional parity lives
under `backend/modules/text_research/tests/quanteda_parity`; the scheduled or
manually dispatched GitHub workflow installs `quanteda` and `jsonlite` before
running the R-specific optional test.

## Required design for a future engine

Any Phase 5 implementation must branch after existing canonical research
inputs, not create R-specific corpus, project, snapshot, or specification
stores:

```text
CorpusSnapshot + DatasetView + AnalysisSpecification
  -> PipelineCompiler + PreparedCorpusArtifact
  -> Python engine | controlled Rscript worker | comparison result
  -> existing artifacts, provenance, AnalysisRun state, and frontend
```

The R worker would need a versioned input manifest, a bounded `Rscript`
subprocess, timeout/resource limits, structured stdout/stderr capture, result
schema validation, artifact/provenance registration, and cleanup. It must use
the existing Celery/job model and preserve the frozen input identities used by
the Python engine.

## Validation status

`Rscript` was unavailable during Phase 7. With `QUANTEDA_R_PARITY=1`, the
optional suite reported 1 passed and 2 skipped; no real quanteda execution was
verified. Python quanteda-analogue parity passed separately (2 tests).
