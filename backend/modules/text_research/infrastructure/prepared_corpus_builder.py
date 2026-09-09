"""Build :class:`PreparedCorpusArtifact` instances from raw unit texts."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from backend.modules.text_research.domain.prepared_corpus import (
    PreparedCorpusArtifact,
    build_prepared_artifact,
    compute_corpus_checksum,
)
from backend.modules.text_research.infrastructure.preprocessing import (
    PreprocessingConfig,
    describe_implementation,
    tokenize,
)


def _tokenize_with_per_unit_language(
    originals: list[str],
    cfg: dict[str, Any],
    *,
    unit_ids: list[str],
    language_override: str | None,
    language_mode: str,
    per_unit_language: bool,
    manual_overrides: dict[str, str] | None,
) -> tuple[list[list[str]], dict[str, Any] | None]:
    """Return token sequences and optional language provenance metadata."""
    from backend.modules.text_research.infrastructure.language_detection import (
        LanguageRouter,
        detect_units,
        detect_with_override,
    )

    language_meta: dict[str, Any] | None = None
    if language_override:
        language_meta = {
            "language": language_override,
            "confidence": 1.0,
            "detector_name": "manual_override",
            "detector_version": "1",
        }
        cfg["language"] = language_override
        return [tokenize(text, cfg) for text in originals], language_meta

    if per_unit_language and originals:
        unit_detections = detect_units(
            originals,
            manual_overrides=manual_overrides,
            unit_ids=unit_ids,
        )
        routed_langs = [
            LanguageRouter().processor_for(item.get("language")) for item in unit_detections
        ]
        unique_langs = {lang for lang in routed_langs if lang != "generic"}
        language_meta = {
            "mode": "per_unit",
            "units": unit_detections,
            "unique_languages": sorted(unique_langs),
        }
        if len(unique_langs) <= 1:
            if unique_langs:
                cfg = dict(cfg)
                cfg["language"] = next(iter(unique_langs))
            return [tokenize(text, cfg) for text in originals], language_meta

        token_sequences: list[list[str]] = []
        for text, lang in zip(originals, routed_langs, strict=True):
            unit_cfg = dict(cfg)
            if lang != "generic":
                unit_cfg["language"] = lang
            token_sequences.append(tokenize(text, unit_cfg))
        return token_sequences, language_meta

    if language_mode == "auto" and originals:
        sample = "\n".join(originals[:20])
        language_meta = detect_with_override(sample, manual_override=None)
        routed = LanguageRouter().processor_for(language_meta.get("language"))
        if routed != "generic":
            cfg = dict(cfg)
            cfg["language"] = routed
        return [tokenize(text, cfg) for text in originals], language_meta

    return [tokenize(text, cfg) for text in originals], language_meta


def prepare_texts(
    texts: list[str],
    config: dict[str, Any] | PreprocessingConfig,
    *,
    unit_ids: list[str] | None = None,
    document_ids: list[str | None] | None = None,
    metadata_by_unit: dict[str, dict[str, Any]] | None = None,
    cleaned_texts: list[str] | None = None,
    provenance: dict[str, Any] | None = None,
    language_mode: str = "manual",
    language_override: str | None = None,
    per_unit_language: bool = False,
    language_overrides_by_unit: dict[str, str] | None = None,
    force_in_memory: bool = False,
) -> PreparedCorpusArtifact:
    """Tokenize ``texts`` without mutating the caller's input strings.

    When ``language_mode="auto"`` and no explicit language is set on the
    profile, run heuristic language detection and route the config language
    field (manual override always wins).
    """
    if isinstance(config, PreprocessingConfig):
        cfg = config.to_dict()
        cfg_obj = config
    else:
        cfg_obj = PreprocessingConfig.from_dict(config)
        cfg = cfg_obj.to_dict()

    # Profile language routing defaults (call-site kwargs still win).
    resolved_language_mode = language_mode
    resolved_per_unit = per_unit_language
    if language_override is None:
        if cfg.get("multilingual") or cfg.get("language_mode") == "per_unit":
            resolved_per_unit = True
            resolved_language_mode = "per_unit"
        elif cfg.get("auto_detect_language") or cfg.get("language_mode") == "auto":
            if resolved_language_mode == "manual":
                resolved_language_mode = "auto"

    originals = list(texts)
    n = len(originals)
    resolved_unit_ids = unit_ids or [f"unit-{index}" for index in range(n)]
    if len(resolved_unit_ids) != n:
        raise ValueError("unit_ids must align 1:1 with texts")

    if not force_in_memory:
        from backend.modules.text_research.infrastructure.out_of_core import (
            prepare_texts_batched,
            should_use_out_of_core,
        )

        if should_use_out_of_core(n):
            return prepare_texts_batched(
                originals,
                config,
                unit_ids=resolved_unit_ids,
                document_ids=document_ids,
                metadata_by_unit=metadata_by_unit,
                cleaned_texts=cleaned_texts,
                provenance=provenance,
                language_mode=resolved_language_mode,
                language_override=language_override,
                per_unit_language=resolved_per_unit,
                language_overrides_by_unit=language_overrides_by_unit,
            )

    token_sequences, language_meta = _tokenize_with_per_unit_language(
        originals,
        cfg,
        unit_ids=resolved_unit_ids,
        language_override=language_override,
        language_mode=resolved_language_mode,
        per_unit_language=resolved_per_unit,
        manual_overrides=language_overrides_by_unit,
    )
    if language_meta and language_meta.get("language") and not resolved_per_unit:
        cfg_obj = PreprocessingConfig.from_dict({**cfg, "language": language_meta["language"]})

    impl_meta = describe_implementation(cfg_obj)

    artifact_provenance = dict(provenance or {})
    artifact_provenance.setdefault("preprocessing_implementation", impl_meta)
    if language_meta is not None:
        artifact_provenance["language_detection"] = language_meta
    if cfg.get("enable_ner"):
        from backend.modules.text_research.infrastructure.nlp_preprocessing import (
            extract_entities_for_provenance,
        )

        artifact_provenance["ner"] = extract_entities_for_provenance(
            originals,
            model_name=str(cfg.get("spacy_model") or "en_core_web_sm"),
            unit_ids=resolved_unit_ids,
        )

    return build_prepared_artifact(
        resolved_unit_ids,
        originals,
        token_sequences,
        cfg,
        metadata_by_unit=metadata_by_unit,
        document_ids=document_ids,
        cleaned_texts=cleaned_texts,
        provenance=artifact_provenance,
        impl_meta=impl_meta,
    )


def prepare_texts_cached(
    texts: list[str],
    config: dict[str, Any] | PreprocessingConfig,
    *,
    corpus_id: str,
    unit_type: str,
    unit_ids: list[str],
    document_ids: list[str | None] | None = None,
    filters: dict[str, Any] | None = None,
    cleaning_profile_hash: str | None = None,
    operation_config: dict[str, Any] | None = None,
    **kwargs: Any,
) -> PreparedCorpusArtifact:
    """Build or reuse an immutable prepared corpus by full scientific inputs.

    Uses the layered stage cache (L1/L2/L3) with stampede protection. Covers
    only deterministic cleaning/tokenization outputs — never fitted
    vectorizers, IDF, supervised selectors, or classifiers.
    """
    from backend.modules.text_research.infrastructure import stage_cache

    resolved_config = config.to_dict() if isinstance(config, PreprocessingConfig) else dict(config)
    input_checksum = compute_corpus_checksum(unit_ids, texts)
    cache_spec = {
        "corpus_id": corpus_id,
        "unit_type": unit_type,
        "filters": filters or {},
        "cleaning_profile_hash": cleaning_profile_hash,
        "preprocessing_config": resolved_config,
        "operation_config": operation_config or {},
        "implementation_version": "prepared-corpus-v1",
    }
    cache_key = stage_cache.stage_cache_key(
        engine_version="prepared-corpus-v1",
        stage_name="prepared_corpus",
        input_checksum=input_checksum,
        spec_hash=hashlib.sha256(
            json.dumps(cache_spec, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest(),
        params=cache_spec,
        preprocessing_config=resolved_config,
    )

    def _factory() -> tuple[dict[str, Any], PreparedCorpusArtifact]:
        prepared = prepare_texts(
            texts,
            config,
            unit_ids=unit_ids,
            document_ids=document_ids,
            **kwargs,
        )
        return (
            {
                "stage_name": "prepared_corpus",
                "corpus_checksum": prepared.corpus_checksum,
                "pipeline_checksum": prepared.pipeline_checksum,
            },
            prepared,
        )

    cached = stage_cache.get_or_compute(cache_key, _factory, payload_format="joblib")
    payload = cached.get("payload")
    if isinstance(payload, PreparedCorpusArtifact):
        return payload
    # Joblib round-trip across processes may lose exact type in edge cases.
    return prepare_texts(
        texts,
        config,
        unit_ids=unit_ids,
        document_ids=document_ids,
        **kwargs,
    )
