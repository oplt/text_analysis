# Quanteda parity: known differences

## Intentional differences

- Python tokenizer uses a Unicode-aware regex (`unicode_regex`) rather than
  quanteda's `tokens()` defaults; punctuation and hyphen handling may diverge
  on edge cases.
- Stopword lists come from language profiles in this codebase, not quanteda's
  built-in `stopwords()` objects.
- Stemming/lemmatization backends differ (Snowball/simplemma vs quanteda's
  optional backends).

## Tolerances

Document acceptable drift tolerances here when R and Python disagree on edge cases.

- Golden Python fixtures in `tiny_corpus.json` are the CI source of truth for
  default preprocessing.
- Optional R parity (`QUANTEDA_R_PARITY=1`) compares regenerated quanteda
  tokens/DFM JSON against the Python fixtures and documents any residual drift
  here rather than failing default CI when R is absent.

## CI note

- Default GitHub Actions jobs do **not** require R or quanteda.
- The optional workflow `.github/workflows/quanteda-parity.yml` runs on
  `workflow_dispatch` (and an optional schedule) inside `rocker/r-ver`,
  installs quanteda, sets `QUANTEDA_R_PARITY=1`, and executes only
  `test_r_parity_optional.py`.
- When R is unavailable locally, that test module skips cleanly.
