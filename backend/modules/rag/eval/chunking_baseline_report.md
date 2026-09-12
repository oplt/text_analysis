# Chunking baseline report (fixture experiment)

Generated from `backend.modules.rag.eval.chunking_experiments` using the offline
corpus fixtures under `eval/fixtures/corpus/` and synthetic ranking lists.

## Defaults retained

| Knob | Current default | Fixture justification |
|------|-----------------|------------------------|
| `chunk_size` | 1000 | Prose fixtures produce a single compact chunk per short document; smaller windows (512/768) increase chunk count without improving synthetic Recall@k/MRR on these fixtures. |
| `chunk_overlap` | 150 | Overlap increases duplicate-token ratio; fixtures do not show ranking gains that would justify a larger overlap. |
| PDF profile | `pdf_page` (≤768/≤96) | Page-aware profile reduces cross-page parent bleed risk; not a live ANN measurement. |
| Table/CSV | atomic, overlap 0 | Row/table records must not be merged with prose overlap. |

## Metrics (synthetic)

The experiment module emits Recall@k and MRR against fixture rankings. These are
structural smoke metrics only — they do **not** replace live pgvector retrieval
benchmarks. Re-run:

```bash
backend/.venv/bin/python -m backend.modules.rag.eval.chunking_experiments
```

## Status

Defaults are **provisionally justified** by offline fixtures. Any production change
to chunk size/overlap requires a live end-to-end retrieval measurement
(NEEDS_LIVE_MEASUREMENT).
