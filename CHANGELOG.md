# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Unified research pipeline architecture: prepared corpus artifacts, pipeline compiler, artifact registry, and analysis task contract.
- Lexical-hash embedding baseline (`HashingEmbeddingProvider`) and optional sentence-transformer provider.
- Property-style pipeline tests for leakage prevention and checksum determinism.
- **P2 platform layer:** content-addressable stage cache, out-of-core prepare batches, Parquet/Arrow (or JSONL/npz) intermediates, dedicated `research_light|cpu|memory|gpu` Celery queues, retry/checkpoint/idempotency policies, analysis plugin registry, workflow recipe composer, model lifecycle states (candidate/approved/deprecated), classifier drift monitoring API, and `text-research` headless CLI.
- **Universal StageRunner** executing compiled analysis plans; quantitative analyses share `PreparedCorpusArtifact`.
- Unit-level language detection; full string+token `TextTransform` pipelines; decomposed `semantic_stack` topic engine; nested grouped CV / embedding classifiers / `custom_utility` thresholds; first-class `PredictionSet`; golden multilingual fixtures; optional R quanteda CI workflow.
- Optional extras: `[nlp]`, `[bertopic]`, `[transformers]`.

### Changed

- Research analyses share a common preprocessing → feature extraction → modeling path via compiled specifications.
- Research Celery tasks route by resource class with policy-driven retries.
