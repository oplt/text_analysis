"""Safe research feature cache for repeated quantitative preprocessing.

Cache identity: corpus snapshot hash + unit type + filters + preprocessing
config + vectorization mode.

Do NOT use this cache for supervised classifier TF-IDF fits — those must
remain fitted on training partitions only to prevent leakage.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import sys
from collections import OrderedDict
from typing import Any

from backend.modules.text_research.infrastructure.preprocessing import (
    PreprocessingConfig,
    tokenize,
)

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_CACHE: OrderedDict[str, Any] = OrderedDict()
_MAX_ENTRIES = 64
_MAX_BYTES = 32 * 1024 * 1024
_CACHE_BYTES = 0
_HITS = 0
_MISSES = 0
_EVICTIONS = 0


def _estimate_size(value: Any) -> int:
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        return size + sum(_estimate_size(key) + _estimate_size(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return size + sum(_estimate_size(item) for item in value)
    return size


def build_cache_key(
    *,
    corpus_id: str,
    unit_type: str,
    unit_ids: list[str],
    unit_hashes: list[str] | None = None,
    config: dict[str, Any] | PreprocessingConfig | None,
    mode: str,
    filters: dict[str, Any] | None = None,
) -> str:
    if isinstance(config, PreprocessingConfig):
        config_payload = config.to_dict()
    else:
        config_payload = PreprocessingConfig.from_dict(config).to_dict()
    # Include persisted text hashes when available so content changes invalidate.
    unit_identity = unit_hashes if unit_hashes is not None else unit_ids
    units_digest = hashlib.sha256(",".join(sorted(unit_identity)).encode("utf-8")).hexdigest()[:24]
    payload = {
        "corpus_id": corpus_id,
        "unit_type": unit_type,
        "units_digest": units_digest,
        "config": config_payload,
        "mode": mode,
        "filters": filters or {},
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def set_cached(key: str, value: Any) -> None:
    global _CACHE_BYTES, _EVICTIONS
    from backend.modules.text_research.infrastructure.out_of_core import (
        should_use_out_of_core,
        spill_token_sequences,
    )

    spill_meta: dict[str, Any] | None = None
    stored: Any = value
    # Spill large tokenized corpora to parquet/JSONL instead of holding them all in RAM.
    if (
        isinstance(value, list)
        and value
        and isinstance(value[0], list)
        and should_use_out_of_core(len(value))
    ):
        try:
            spill_meta = spill_token_sequences(key, value)
            stored = {"__spill__": spill_meta}
        except Exception:
            logger.exception("feature_cache spill failed; keeping in-memory entry")
            spill_meta = None
            stored = value

    with _LOCK:
        previous = _CACHE.pop(key, None)
        if previous is not None:
            _CACHE_BYTES -= _estimate_size(previous)
        _CACHE[key] = stored
        _CACHE_BYTES += _estimate_size(stored)
        _CACHE.move_to_end(key)
        while len(_CACHE) > _MAX_ENTRIES or _CACHE_BYTES > _MAX_BYTES:
            _, evicted = _CACHE.popitem(last=False)
            _CACHE_BYTES -= _estimate_size(evicted)
            _EVICTIONS += 1


def get_cached(key: str) -> Any | None:
    global _HITS, _MISSES
    with _LOCK:
        value = _CACHE.get(key)
        if value is None:
            _MISSES += 1
            return None
        _HITS += 1
        _CACHE.move_to_end(key)
        stored = value

    if isinstance(stored, dict) and "__spill__" in stored:
        from backend.modules.text_research.infrastructure.out_of_core import (
            load_spilled_token_sequences,
        )

        try:
            return load_spilled_token_sequences(stored["__spill__"])
        except Exception:
            logger.exception("feature_cache spill load failed for key=%s", key)
            return None
    return stored


def clear_cache() -> None:
    global _CACHE_BYTES
    with _LOCK:
        _CACHE.clear()
        _CACHE_BYTES = 0


def invalidate_cache(key: str | None = None) -> None:
    """Invalidate one entry or the entire in-process quantitative cache."""
    global _CACHE_BYTES
    with _LOCK:
        if key is None:
            _CACHE.clear()
            _CACHE_BYTES = 0
            return
        value = _CACHE.pop(key, None)
        if value is not None:
            _CACHE_BYTES -= _estimate_size(value)


def cache_metrics() -> dict[str, int]:
    with _LOCK:
        return {"entries": len(_CACHE), "bytes": _CACHE_BYTES, "hits": _HITS, "misses": _MISSES, "evictions": _EVICTIONS}


def get_or_tokenize(
    *,
    corpus_id: str,
    unit_type: str,
    unit_ids: list[str],
    texts: list[str],
    config: dict[str, Any] | PreprocessingConfig | None,
    filters: dict[str, Any] | None = None,
) -> list[list[str]]:
    """Return tokenized units, using the in-process cache when available."""
    key = build_cache_key(
        corpus_id=corpus_id,
        unit_type=unit_type,
        unit_ids=unit_ids,
        config=config,
        mode="tokens",
        filters=filters,
    )
    cached = get_cached(key)
    if cached is not None:
        return cached
    cfg = config.to_dict() if isinstance(config, PreprocessingConfig) else config
    tokenized = [tokenize(text, cfg) for text in texts]
    set_cached(key, tokenized)
    return tokenized
