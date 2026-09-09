# Text Research Engineering Report (§76)

Date: 2026-09-09  
Git HEAD (at report time): `3523411`  
Scope: `prompt.txt` sections **21–76** (building on prior §§1–20 work).

## Summary

The `text_research` backend is upgraded into a **generic** research-grade text platform: canonical corpus integrity, quanteda-analogue quantitative path, generic annotation/sampling/reliability, leakage-aware classification, unsupervised extras, comparative analysis over user metadata, and reproducibility exports. Domain-specific labels, dictionaries, and metadata categories are not hardcoded.

---

## Changed / created files

Paths relative to repo root (working tree at report time):

### Migrations
- `backend/alembic/versions/a1b2c3d4e5f6_add_research_canonical_sources.py`
- `backend/alembic/versions/b2c3d4e5f6a7_text_unit_source_provenance.py`
- `backend/alembic/versions/c3d4e5f6a7b8_add_cleaning_profiles.py`

### Domain / API / application
- `backend/modules/text_research/domain/enums.py`
- `backend/modules/text_research/domain/models.py`
- `backend/modules/text_research/domain/analysis_specification.py` *(new)*
- `backend/modules/text_research/api/routes.py`
- `backend/modules/text_research/api/schemas.py`
- `backend/modules/text_research/application/*` (services updated/added: cleaning, ingestion QA, sampling, quantitative, classification, robustness, comparative, export, statistical modeling, measurement validation, …)

### Infrastructure engines (new / extended)
- `canonical_text.py`, `ingestion_qa.py`, `document_cleaning.py`, `language_processing.py`
- `weighting.py`, `kwic.py`, `dictionary_matcher.py`, `keyness.py`, `collocation.py`, `association_network.py`
- `similarity.py`, `duplicate_detection.py`, `sampling.py`, `validation_splits.py`
- `clustering.py`, `dimensionality.py`, `readability.py`, `embeddings.py`, `ner.py`, `linguistic_features.py`
- `statistical_modeling.py`, `measurement_validation.py`
- plus extensions to `preprocessing.py`, `quantitative.py`, `classifiers.py`, `topic_models.py`, `reliability.py`, `feature_cache.py`, …

### Tests
- Many under `backend/modules/text_research/tests/`
- Quanteda-analogue suite: `tests/quanteda_parity/` (+ fixtures)

### Docs / config / frontend touchpoints
- `prompt.txt`, `docs/text-research.md`, `docs/text-research-engineering-report.md` (this file)
- `backend/pyproject.toml` (adds `statsmodels`)
- Frontend research views/API types updated for additive fields where needed

---

## Architecture changes

| Abstraction | Role |
|---|---|
| **Canonical source document** | Research text from dedicated extraction/checksum path — never reconstructed from RAG chunks |
| **Provenance on text units** | Source/page/offset (where available) for KWIC and audit |
| **Cleaning + preprocessing profiles** | Explicit, versioned transforms before tokenization |
| **Standardized quantitative pipeline** | Corpus → tokens → preprocessing → DFM → trim → weighting → analysis |
| **`AnalysisSpecification`** | Shared config vocabulary (corpus, features, model, validation, seed) |
| **Grouped validation helpers** | Holdout / CV / leave-one-group-out / temporal — group by document or user metadata |
| **Classifier pipelines** | sklearn `Pipeline`; vectorizer/calibrator fit on train(/val) only |
| **Optional families** | Embeddings / NER / spaCy linguistic features fail honestly when unavailable |
| **Feature cache** | Quantitative tokenization cache only — not for supervised TF-IDF fits |
| **AnalysisRun + TrainedModel** | DB-backed experiment tracking (no MLflow) |

Layering preserved: `router → application/service → repository/infrastructure`. Domain does not import infrastructure engines.

---

## Database changes

| Migration | Purpose |
|---|---|
| `a1b2c3d4e5f6` | Research canonical sources (text, checksums, parser metadata) |
| `b2c3d4e5f6a7` | Text-unit source provenance fields |
| `c3d4e5f6a7b8` | Cleaning profiles |

Later capabilities (similarity, clustering, statistical model, triangulation, richer snapshots) prefer **additive JSON** on `AnalysisRun` / snapshot payloads rather than new tables.

---

## Implemented analyses (by family)

### Corpus integrity
Canonical extraction, checksums, ingestion QA (incl. near-duplicate detection), cleaning profiles.

### Preprocessing
Unicode/ftfy options, language-aware stemming/lemmatization architecture, sentence segmentation, profile provenance.

### Corpus statistics / frequencies / DFM
Corpus stats, term & n-gram frequencies, DFM with trim + weighting (`count`, `binary`, `tf`, `tfidf`, `sublinear_tf`, `log_count`, `bm25`).

### KWIC / dictionaries / keyness / co-occurrence
KWIC (word/phrase/regex/wildcard/lemma), hierarchical user dictionaries, keyness (G²/χ²/Fisher + FDR + effect sizes), collocation metrics + association networks.

### Similarity / duplicates
TF-IDF cosine, Jaccard; embedding cosine only if vectors supplied; exact/normalized/lexical/MinHash duplicates.

### Annotation / reliability / freeze
Generic codebooks/labels; stratified sampling; Fleiss’ κ (+ metadata); frozen dataset snapshots with hashes.

### Classification / validation
Binary/multiclass/multilabel (inferred or explicit); NB / Complement NB / SGD / LR / SVM; word+char features; hyperparameter search; threshold tuning; imbalance (`class_weight`); expanded metrics; bootstrap CIs; calibration; coefficient explainability; robustness (group/temporal/transfer).

### Topic modeling / clustering / dim-red / readability
LDA/NMF with K-sweep & seed stability; KMeans(+SVD); TruncatedSVD/PCA; readability/style features.

### Comparative / statistical / triangulation
Prevalence by **any** document metadata field; user-specified OLS (+ optional logistic via statsmodels); measurement comparison (agreement/correlation/confusion) without auto-equating concepts.

### Reproducibility
Manifest export with corpus/canonical checksums, profiles, snapshots, models, runs, library versions, git SHA; quanteda load script helper; Python reference parity fixtures.

---

## Tests

```bash
cd /home/polat/Desktop/Projects/text_analysis
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  backend/.venv/bin/pytest backend/modules/text_research/tests/ -q
```

| Result | Count |
|---|---|
| **Passed** | **351** |
| Failed | 0 |
| Skipped | 0 |
| Warnings | 2 (passlib `crypt` deprecation; NMF sqrt warning on tiny synthetic data) |

---

## Remaining limitations

Be explicit:

1. **Embeddings** — provider stub (`UnavailableEmbeddingProvider`); no default transformer backend; RAG vectors must not be used as research text.
2. **NER / POS / linguistic features** — optional spaCy; honest failure when model missing.
3. **UMAP / BERTopic / transformers** — not required; not shipped as core.
4. **Weighted ordinal κ** — no end-to-end ordinal label scale yet.
5. **Annotation hierarchy** — parent/child label graph not fully first-class in API.
6. **Quanteda parity** — Python reference fixtures + documented differences; R not in CI; external R comparison is optional research work.
7. **Statistical layer** — OLS core always available; logistic needs `statsmodels`; Poisson/NegBin not yet exposed.
8. **Triangulation** — compares caller-supplied series; does not auto-join annotation/dictionary/topic tables into one UI wizard.
9. **Contextual joins** — still document `country` / `publication_year` helpers for external indicators (separate from comparative `group_by`).
10. **Large-corpus streaming** — sparse DFM + cache + async runs present; full out-of-core sharding is partial.
11. **Frontend** — additive API coverage; not every new analysis has a polished dedicated screen.

---

## Generic example workflow

`upload documents`
→ `create corpus`
→ `canonicalize`
→ `run QA`
→ `add/import metadata`
→ `choose research unit`
→ `segment`
→ `create preprocessing profile`
→ `inspect preprocessing`
→ `build DFM`
→ `run frequencies / KWIC / keyness / dictionary / co-occurrence`
→ `create user-defined codebook`
→ `sample units`
→ `annotate`
→ `calculate reliability`
→ `adjudicate`
→ `freeze dataset`
→ `select classification task`
→ `train baseline models`
→ `perform grouped/nested validation`
→ `evaluate`
→ `predict remaining corpus`
→ `run topic modeling`
→ `run similarity/clustering if desired`
→ `compare results using user-selected metadata`
→ `export reproducibility manifest`

The platform is intended as a **general-purpose** computational text-analysis environment: changing substantive concepts, metadata fields, labels, or dictionaries should not require source-code changes.

---

## Follow-up: architecture gap closure (2026-09-09)

See `docs/text-research-gap-implementation-status.md`.

Added: `PreparedCorpusArtifact` shared prepare path; AnalysisSpecification v2 +
PipelineCompiler; artifact registry; stratified SplitPlanner; language detection;
TextTransform compose; Hungarian topic stability; topic `group_by`; threshold
objectives + abstention; error analysis; hashing embeddings + embedding classifier;
optional BERTopic/spaCy engines; AnalysisTask contract; pipeline property tests;
MIT LICENSE + community docs.

