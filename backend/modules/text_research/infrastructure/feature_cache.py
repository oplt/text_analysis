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


def get_cached(key: str) -> Any | None:
    with _LOCK:
        value = _CACHE.get(key)
        if value is None:
            return None
        _CACHE.move_to_end(key)
        return value


def set_cached(key: str, value: Any) -> None:
    with _LOCK:
        _CACHE[key] = value
        _CACHE.move_to_end(key)
        while len(_CACHE) > _MAX_ENTRIES:
            _CACHE.popitem(last=False)


def clear_cache() -> None:
    with _LOCK:
        _CACHE.clear()


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
