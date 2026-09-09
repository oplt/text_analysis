"""Out-of-core / large-corpus batch processing helpers.

Prefer PyArrow/Parquet intermediates, scipy CSR, HashingVectorizer for
exploratory huge corpora, MiniBatchKMeans, and chunked DB writes. Never
densify a full large sparse matrix — only small previews.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, TypeVar

from scipy import sparse

from backend.modules.text_research.domain.prepared_corpus import (
    PreparedCorpusArtifact,
    build_prepared_artifact,
)
from backend.modules.text_research.infrastructure.preprocessing import (
    PreprocessingConfig,
    describe_implementation,
    tokenize,
)

T = TypeVar("T")

DEFAULT_BATCH_SIZE = 500
DEFAULT_HASHING_N_FEATURES = 2**18
# Cap in-memory COO export rows so DFM results stay lean for large corpora.
DEFAULT_SPARSE_PAYLOAD_NNZ_CAP = 250_000


def should_use_out_of_core(n_units: int, threshold: int | None = None) -> bool:
    """Return True when corpus size meets the large-corpus threshold."""
    if threshold is None:
        try:
            from backend.core.config import settings

            threshold = settings.RESEARCH_LARGE_CORPUS_DOCUMENT_THRESHOLD
        except Exception:
            threshold = 50
    return n_units >= threshold


def resolve_batch_size(n_units: int, batch_size: int | None = None) -> int:
    """Pick a batch size; default scales gently with corpus size."""
    if batch_size is not None and batch_size > 0:
        return int(batch_size)
    if n_units >= 10_000:
        return 2_000
    if n_units >= 1_000:
        return 1_000
    return DEFAULT_BATCH_SIZE


def iter_item_batches(items: Sequence[T], batch_size: int = DEFAULT_BATCH_SIZE) -> Iterator[Sequence[T]]:
    """Yield fixed-size slices of any sequence (DB bulk, tokenization, etc.)."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def iter_text_batches(
    texts: list[str],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    unit_ids: list[str] | None = None,
) -> Iterator[tuple[list[str], list[str]]]:
    """Yield ``(batch_ids, batch_texts)`` slices covering all units."""
    n = len(texts)
    resolved_unit_ids = unit_ids or [f"unit-{index}" for index in range(n)]
    if len(resolved_unit_ids) != n:
        raise ValueError("unit_ids must align 1:1 with texts")

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        yield resolved_unit_ids[start:end], texts[start:end]


def polars_available() -> bool:
    try:
        import polars  # noqa: F401

        return True
    except ImportError:
        return False


def pyarrow_available() -> bool:
    try:
        import pyarrow  # noqa: F401

        return True
    except ImportError:
        return False


def ensure_csr(matrix: Any) -> sparse.csr_matrix:
    """Return a CSR view/copy; never densify."""
    if sparse.issparse(matrix):
        return matrix.tocsr()
    return sparse.csr_matrix(matrix)


def should_force_sparse_only(
    n_rows: int,
    n_cols: int,
    *,
    dense_export_limit: int = 5000,
    force_sparse_only: bool = False,
) -> bool:
    """True when a full dense export would be too large or was explicitly requested."""
    if force_sparse_only:
        return True
    cells = int(n_rows) * int(n_cols)
    if cells <= 0:
        return True
    if cells > dense_export_limit:
        return True
    return should_use_out_of_core(n_rows)


def dense_preview(
    matrix: Any,
    *,
    max_rows: int = 10,
    max_cols: int = 10,
) -> list[list[float]]:
    """Densify only a tiny corner of a sparse matrix for UI previews."""
    if matrix is None:
        return []
    csr = ensure_csr(matrix)
    n_rows = min(max_rows, int(csr.shape[0]))
    n_cols = min(max_cols, int(csr.shape[1]))
    if n_rows == 0 or n_cols == 0:
        return []
    return csr[:n_rows, :n_cols].toarray().tolist()


def recommend_clustering_algorithm(n_units: int, requested: str = "auto") -> str:
    """Prefer MiniBatchKMeans for large corpora; honour explicit requests."""
    algo = (requested or "auto").lower()
    if algo in {"auto", "default"}:
        return "minibatch_kmeans" if should_use_out_of_core(n_units) else "kmeans"
    return algo


def recommend_vectorizer_mode(n_units: int, requested: str | None = None) -> str:
    """Prefer HashingVectorizer for huge exploratory corpora."""
    mode = (requested or "auto").lower()
    if mode in {"auto", "default"}:
        return "hashing" if should_use_out_of_core(n_units) else "count"
    if mode not in {"count", "tfidf", "hashing"}:
        raise ValueError(f"Unsupported vectorizer mode {requested!r}")
    return mode


def build_hashing_matrix(
    texts: Sequence[str],
    *,
    n_features: int = DEFAULT_HASHING_N_FEATURES,
    batch_size: int | None = None,
    config: dict[str, Any] | PreprocessingConfig | None = None,
    alternate_sign: bool = True,
    norm: str | None = None,
) -> sparse.csr_matrix:
    """Fit-free HashingVectorizer transform in batches; return stacked CSR.

    Vocabulary is never materialized — suitable for exploratory huge corpora
    where CountVectorizer memory would explode.
    """
    from backend.modules.text_research.infrastructure.preprocessing import (
        build_hashing_vectorizer,
    )

    if not texts:
        return sparse.csr_matrix((0, n_features))

    vectorizer = build_hashing_vectorizer(
        config,
        n_features=n_features,
        alternate_sign=alternate_sign,
        norm=norm,
    )
    size = resolve_batch_size(len(texts), batch_size)
    blocks: list[sparse.csr_matrix] = []
    for batch in iter_item_batches(list(texts), size):
        block = vectorizer.transform(list(batch))
        blocks.append(ensure_csr(block))
    if len(blocks) == 1:
        return blocks[0]
    return sparse.vstack(blocks, format="csr")


def sparse_payload_from_matrix(
    matrix: Any,
    *,
    nnz_cap: int = DEFAULT_SPARSE_PAYLOAD_NNZ_CAP,
) -> dict[str, Any]:
    """Serialize sparse matrix as COO lists, truncating nnz for large matrices."""
    csr = ensure_csr(matrix)
    coo = csr.tocoo()
    nnz = int(coo.nnz)
    truncated = nnz > nnz_cap
    limit = nnz_cap if truncated else nnz
    payload: dict[str, Any] = {
        "format": "coo",
        "row": coo.row[:limit].tolist(),
        "col": coo.col[:limit].tolist(),
        "data": [float(v) for v in coo.data[:limit].tolist()],
        "shape": [int(csr.shape[0]), int(csr.shape[1])],
        "nnz": nnz,
        "truncated": truncated,
    }
    if truncated:
        payload["nnz_exported"] = limit
        payload["note"] = (
            f"COO payload truncated to {limit} of {nnz} nonzeros; "
            "use parquet/npz artifact for the full sparse matrix."
        )
    return payload


def spill_sparse_matrix_artifact(
    path_stem: str | Path,
    matrix: Any,
    feature_names: list[str] | None = None,
) -> dict[str, Any]:
    """Persist CSR/COO via parquet_artifacts (Parquet COO or npz fallback)."""
    from backend.modules.text_research.infrastructure.parquet_artifacts import save_sparse_matrix

    return save_sparse_matrix(path_stem, ensure_csr(matrix), feature_names or [])


def feature_cache_spill_root() -> Path:
    try:
        from backend.core.config import settings

        root = Path(settings.RESEARCH_ARTIFACT_DIR or "var/research_artifacts")
    except Exception:
        root = Path("var/research_artifacts")
    path = root / "feature_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def spill_token_sequences(
    key: str,
    token_sequences: list[list[str]],
) -> dict[str, Any]:
    """Persist tokenized units as parquet/JSONL for large feature-cache entries."""
    from backend.modules.text_research.infrastructure.parquet_artifacts import save_unit_table

    rows = [
        {"unit_index": index, "tokens": " ".join(tokens), "token_count": len(tokens)}
        for index, tokens in enumerate(token_sequences)
    ]
    meta = save_unit_table(feature_cache_spill_root() / key, rows)
    meta["kind"] = "token_sequences"
    meta["unit_count"] = len(token_sequences)
    return meta


def load_spilled_token_sequences(meta: dict[str, Any]) -> list[list[str]]:
    from backend.modules.text_research.infrastructure.parquet_artifacts import load_unit_table

    rows = load_unit_table(meta["path"])
    rows_sorted = sorted(rows, key=lambda row: int(row.get("unit_index", 0)))
    return [str(row.get("tokens") or "").split() for row in rows_sorted]


def duplicate_methods_for_scale(
    methods: Sequence[str],
    n_items: int,
    *,
    force_lexical: bool = False,
) -> tuple[list[str], list[str]]:
    """Drop O(n²) lexical pairwise when corpus is large unless forced.

    Returns ``(resolved_methods, notes)``.
    """
    resolved = list(methods)
    notes: list[str] = []
    if should_use_out_of_core(n_items) and "lexical" in resolved and not force_lexical:
        resolved = [m for m in resolved if m != "lexical"]
        notes.append(
            "Skipped lexical pairwise near-duplicate scan on large corpus "
            f"(n={n_items}); use minhash or set force_lexical=true."
        )
        if "minhash" not in resolved:
            resolved.append("minhash")
            notes.append("Auto-enabled minhash for scalable near-duplicate candidates.")
    return resolved, notes


def streaming_token_counts(
    texts: list[str],
    config: dict[str, Any] | PreprocessingConfig,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Counter[str]:
    """Memory-light token frequency estimate by merging per-batch counters."""
    cfg = config.to_dict() if isinstance(config, PreprocessingConfig) else dict(config)
    merged: Counter[str] = Counter()
    for _, batch_texts in iter_text_batches(texts, batch_size=batch_size):
        for text in batch_texts:
            merged.update(tokenize(text, cfg))
    return merged


def prepare_texts_batched(
    texts: list[str],
    config: dict[str, Any] | PreprocessingConfig,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    unit_ids: list[str] | None = None,
    document_ids: list[str | None] | None = None,
    metadata_by_unit: dict[str, dict[str, Any]] | None = None,
    cleaned_texts: list[str] | None = None,
    provenance: dict[str, Any] | None = None,
    language_mode: str = "manual",
    language_override: str | None = None,
    per_unit_language: bool = False,
    language_overrides_by_unit: dict[str, str] | None = None,
) -> PreparedCorpusArtifact:
    """Tokenize in batches and assemble a single prepared corpus artifact."""
    if isinstance(config, PreprocessingConfig):
        cfg = config.to_dict()
        cfg_obj = config
    else:
        cfg_obj = PreprocessingConfig.from_dict(config)
        cfg = cfg_obj.to_dict()

    originals = list(texts)
    n = len(originals)
    resolved_unit_ids = unit_ids or [f"unit-{index}" for index in range(n)]
    if len(resolved_unit_ids) != n:
        raise ValueError("unit_ids must align 1:1 with texts")

    from backend.modules.text_research.infrastructure.prepared_corpus_builder import (
        _tokenize_with_per_unit_language,
    )

    language_meta: dict[str, Any] | None = None
    size = resolve_batch_size(n, batch_size)
    if per_unit_language or language_override or language_mode == "auto":
        token_sequences, language_meta = _tokenize_with_per_unit_language(
            originals,
            cfg,
            unit_ids=resolved_unit_ids,
            language_override=language_override,
            language_mode=language_mode,
            per_unit_language=per_unit_language,
            manual_overrides=language_overrides_by_unit,
        )
        if language_meta and language_meta.get("language") and not per_unit_language:
            cfg_obj = PreprocessingConfig.from_dict({**cfg, "language": language_meta["language"]})
    else:
        token_sequences = []
        for _, batch_texts in iter_text_batches(
            originals,
            batch_size=size,
            unit_ids=resolved_unit_ids,
        ):
            token_sequences.extend(tokenize(text, cfg) for text in batch_texts)

    impl_meta = describe_implementation(cfg_obj)

    artifact_provenance = dict(provenance or {})
    artifact_provenance.setdefault("preprocessing_implementation", impl_meta)
    artifact_provenance["out_of_core"] = {
        "batch_size": size,
        "pyarrow": pyarrow_available(),
        "polars": polars_available(),
    }
    if language_meta is not None:
        artifact_provenance["language_detection"] = language_meta

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
