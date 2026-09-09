"""Layered content-addressable cache for research pipeline stage outputs.

Layers
------
L1 — small per-process LRU (hot metadata + small payloads)
L2 — Redis metadata, computation status, locks, and L3 pointers
L3 — shared artifact storage (S3/MinIO when configured, else shared local dir)

Cache identity is content-addressed over engine version, stage name, corpus
snapshot checksum, specification hash, parameters, and preprocessing config.

Stampede protection uses Redis locks keyed ``research:lock:{computation_hash}``.

Methodological safeguard: never cache leakage-sensitive supervised fits
(train-fitted TF-IDF / feature selectors). Those must stay partition-local.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import shutil
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from backend.modules.text_research.infrastructure import artifact_registry
from backend.modules.text_research.infrastructure.pipeline_compiler import computation_identity

logger = logging.getLogger(__name__)

PayloadFormat = str  # "json" | "joblib" | "npz"

# Stages that must never be shared globally (train/test leakage risk).
LEAKAGE_SENSITIVE_STAGES = frozenset(
    {
        "supervised_vectorizer",
        "supervised_feature_selector",
        "train_fitted_tfidf",
        "classifier_train_features",
        "supervised_feature_matrix",
    }
)

REDIS_META_PREFIX = "research:stage:meta:"
REDIS_STATUS_PREFIX = "research:stage:status:"
REDIS_LOCK_PREFIX = "research:lock:"

_RELEASE_LOCK_LUA = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
"""

# --- L1 process LRU ---------------------------------------------------------

_L1_LOCK = threading.RLock()
_L1: OrderedDict[str, dict[str, Any]] = OrderedDict()
_L1_BYTES = 0
_PROCESS_LOCKS: dict[str, threading.Lock] = {}
_PROCESS_LOCKS_GUARD = threading.Lock()

_sync_redis: Any | None = None
_sync_redis_failed = False


def _settings():
    from backend.core.config import settings

    return settings


def _l1_limits() -> tuple[int, int]:
    try:
        s = _settings()
        return (
            int(getattr(s, "RESEARCH_STAGE_L1_MAX_ENTRIES", 32)),
            int(getattr(s, "RESEARCH_STAGE_L1_MAX_BYTES", 16 * 1024 * 1024)),
        )
    except Exception:
        return 32, 16 * 1024 * 1024


def _ttl_seconds() -> int:
    try:
        return int(_settings().RESEARCH_STAGE_CACHE_TTL_SECONDS)
    except Exception:
        return 604800


def _lock_ttl() -> int:
    try:
        return int(_settings().RESEARCH_STAGE_LOCK_TTL_SECONDS)
    except Exception:
        return 120


def _lock_wait() -> float:
    try:
        return float(_settings().RESEARCH_STAGE_LOCK_WAIT_SECONDS)
    except Exception:
        return 60.0


def _estimate_size(value: Any) -> int:
    try:
        if isinstance(value, dict | list | tuple | str | bytes):
            return len(json.dumps(value, default=str).encode("utf-8"))
    except Exception:
        pass
    return 1024


def _l1_get(key: str) -> dict[str, Any] | None:
    with _L1_LOCK:
        entry = _L1.get(key)
        if entry is None:
            return None
        _L1.move_to_end(key)
        return dict(entry)


def _l1_put(key: str, meta: dict[str, Any]) -> None:
    global _L1_BYTES
    max_entries, max_bytes = _l1_limits()
    size = _estimate_size(meta)
    store = dict(meta)
    if size > max_bytes // 4 and "payload" in store:
        store = {k: v for k, v in store.items() if k != "payload"}
        size = _estimate_size(store)
    with _L1_LOCK:
        previous = _L1.pop(key, None)
        if previous is not None:
            _L1_BYTES -= _estimate_size(previous)
        _L1[key] = store
        _L1_BYTES += size
        _L1.move_to_end(key)
        while _L1 and (len(_L1) > max_entries or _L1_BYTES > max_bytes):
            _, evicted = _L1.popitem(last=False)
            _L1_BYTES -= _estimate_size(evicted)


def _l1_delete(key: str | None = None) -> None:
    global _L1_BYTES
    with _L1_LOCK:
        if key is None:
            _L1.clear()
            _L1_BYTES = 0
            return
        previous = _L1.pop(key, None)
        if previous is not None:
            _L1_BYTES -= _estimate_size(previous)


def _process_lock(key: str) -> threading.Lock:
    with _PROCESS_LOCKS_GUARD:
        lock = _PROCESS_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _PROCESS_LOCKS[key] = lock
        return lock


# --- L2 Redis ---------------------------------------------------------------


def _redis_enabled() -> bool:
    try:
        return bool(_settings().CACHE_ENABLED)
    except Exception:
        return False


def _get_sync_redis() -> Any | None:
    global _sync_redis, _sync_redis_failed
    if not _redis_enabled() or _sync_redis_failed:
        return None
    if _sync_redis is not None:
        return _sync_redis
    try:
        import redis

        _sync_redis = redis.from_url(_settings().REDIS_URL, decode_responses=True)
        _sync_redis.ping()
        return _sync_redis
    except Exception:
        logger.debug("research stage cache: Redis unavailable", exc_info=True)
        _sync_redis_failed = True
        _sync_redis = None
        return None


def reset_redis_client_for_tests() -> None:
    """Test helper: drop cached Redis client / failure latch."""
    global _sync_redis, _sync_redis_failed
    _sync_redis = None
    _sync_redis_failed = False


def _meta_redis_key(key: str) -> str:
    return f"{REDIS_META_PREFIX}{key}"


def _status_redis_key(key: str) -> str:
    return f"{REDIS_STATUS_PREFIX}{key}"


def _lock_redis_key(key: str) -> str:
    return f"{REDIS_LOCK_PREFIX}{key}"


def _redis_get_meta(key: str) -> dict[str, Any] | None:
    client = _get_sync_redis()
    if client is None:
        return None
    try:
        raw = client.get(_meta_redis_key(key))
        if raw is None:
            return None
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except Exception:
        logger.debug("stage cache Redis get failed key=%s", key, exc_info=True)
        return None


def _redis_set_meta(key: str, meta: dict[str, Any]) -> None:
    client = _get_sync_redis()
    if client is None:
        return
    pointer = {k: v for k, v in meta.items() if k != "payload"}
    try:
        client.setex(
            _meta_redis_key(key),
            _ttl_seconds(),
            json.dumps(pointer, sort_keys=True, ensure_ascii=True, default=str),
        )
        client.setex(_status_redis_key(key), _ttl_seconds(), "ready")
    except Exception:
        logger.debug("stage cache Redis set failed key=%s", key, exc_info=True)


def _redis_set_status(key: str, status: str) -> None:
    client = _get_sync_redis()
    if client is None:
        return
    try:
        client.setex(_status_redis_key(key), _ttl_seconds(), status)
    except Exception:
        logger.debug("stage cache Redis status failed key=%s", key, exc_info=True)


def _redis_get_status(key: str) -> str | None:
    client = _get_sync_redis()
    if client is None:
        return None
    try:
        value = client.get(_status_redis_key(key))
        return str(value) if value is not None else None
    except Exception:
        return None


def _redis_delete(key: str) -> None:
    client = _get_sync_redis()
    if client is None:
        return
    try:
        client.delete(_meta_redis_key(key), _status_redis_key(key), _lock_redis_key(key))
    except Exception:
        logger.debug("stage cache Redis delete failed key=%s", key, exc_info=True)


@contextmanager
def distributed_lock(computation_hash: str, *, ttl: int | None = None, wait: float | None = None):
    """Acquire ``research:lock:{hash}``; falls back to process lock if Redis down.

    Yields ``True`` when this caller holds the exclusive compute token,
    ``False`` when the wait timed out (caller may still double-check cache).
    Stale locks expire via Redis TTL.
    """
    lock_ttl = _lock_ttl() if ttl is None else ttl
    wait_s = _lock_wait() if wait is None else wait
    token = str(uuid.uuid4())
    redis_key = _lock_redis_key(computation_hash)
    client = _get_sync_redis()
    acquired_redis = False
    process_lock = _process_lock(computation_hash)
    got_process = process_lock.acquire(timeout=wait_s)

    if client is not None and got_process:
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            try:
                if client.set(redis_key, token, nx=True, ex=lock_ttl):
                    acquired_redis = True
                    break
                status = _redis_get_status(computation_hash)
                if status == "ready" and has_stage(computation_hash):
                    break
            except Exception:
                logger.debug("stage cache lock acquire failed", exc_info=True)
                break
            time.sleep(0.05)

    held = bool(got_process and (acquired_redis or client is None))
    try:
        yield held
    finally:
        if acquired_redis and client is not None:
            try:
                client.eval(_RELEASE_LOCK_LUA, 1, redis_key, token)
            except Exception:
                logger.debug("stage cache lock release failed", exc_info=True)
        if got_process:
            process_lock.release()


# --- L3 shared storage ------------------------------------------------------


def stage_cache_key(
    *,
    engine_version: str,
    stage_name: str,
    input_checksum: str,
    spec_hash: str,
    params: dict[str, Any] | None = None,
    preprocessing_config: dict[str, Any] | None = None,
) -> str:
    """Stable sha256 key over scientifically relevant stage identity inputs."""
    if stage_name in LEAKAGE_SENSITIVE_STAGES or stage_name.startswith("supervised_"):
        raise ValueError(
            f"Refusing to build a shared cache key for leakage-sensitive stage {stage_name!r}"
        )
    merged_params = dict(params or {})
    if preprocessing_config is not None:
        merged_params["preprocessing_config"] = preprocessing_config
    payload = json.dumps(
        {
            "engine_version": engine_version,
            "stage_name": stage_name,
            "input_checksum": input_checksum,
            "spec_hash": spec_hash,
            "params": merged_params,
        },
        sort_keys=True,
        ensure_ascii=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_identity_lookup(
    spec_hash: str,
    corpus_snapshot_hash: str,
    engine_version: str,
) -> str:
    """Wrap :func:`computation_identity` for prepared-corpus stage lookups."""
    return computation_identity(spec_hash, corpus_snapshot_hash, engine_version)


def _stage_cache_root() -> Path:
    env_override = os.environ.get("RESEARCH_ARTIFACT_DIR")
    if env_override:
        return Path(env_override) / "stage_cache"

    try:
        configured = getattr(_settings(), "RESEARCH_ARTIFACT_DIR", "") or ""
        if configured:
            return Path(configured) / "stage_cache"
    except Exception:
        pass

    backend_root = Path(__file__).resolve().parents[3]
    return backend_root / "var" / "research_artifacts" / "stage_cache"


def _entry_dir(key: str) -> Path:
    return _stage_cache_root() / key


def _meta_path(key: str) -> Path:
    return _entry_dir(key) / "meta.json"


def _payload_extension(payload_format: str) -> str:
    if payload_format == "json":
        return ".json"
    if payload_format == "joblib":
        return ".joblib"
    if payload_format == "npz":
        return ".npz"
    raise ValueError(f"Unsupported payload_format {payload_format!r}")


def _object_storage_configured() -> bool:
    try:
        from backend.core.storage import object_storage

        return object_storage.is_configured
    except Exception:
        return False


def _serialize_payload(payload: Any, payload_format: str) -> bytes:
    if payload_format == "json":
        return json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")
    if payload_format == "joblib":
        buf = io.BytesIO()
        joblib.dump(payload, buf)
        return buf.getvalue()
    if payload_format == "npz":
        buf = io.BytesIO()
        if isinstance(payload, dict):
            np.savez_compressed(buf, **payload)
        else:
            np.savez_compressed(buf, data=payload)
        return buf.getvalue()
    raise ValueError(f"Unsupported payload_format {payload_format!r}")


def _deserialize_payload(data: bytes, payload_format: str) -> Any:
    if payload_format == "json":
        return json.loads(data.decode("utf-8"))
    if payload_format == "joblib":
        return joblib.load(io.BytesIO(data))
    if payload_format == "npz":
        loaded = np.load(io.BytesIO(data), allow_pickle=False)
        return {name: loaded[name] for name in loaded.files}
    raise ValueError(f"Unsupported payload_format {payload_format!r}")


def _load_payload_from_disk(path: Path, payload_format: str) -> Any:
    if payload_format == "json":
        return json.loads(path.read_text(encoding="utf-8"))
    if payload_format == "joblib":
        return joblib.load(path)
    if payload_format == "npz":
        loaded = np.load(path, allow_pickle=False)
        return {name: loaded[name] for name in loaded.files}
    raise ValueError(f"Unsupported payload_format {payload_format!r}")


def _load_payload_from_object(object_key: str, payload_format: str) -> Any:
    from backend.core.storage import object_storage

    body = object_storage.download_bytes_sync(object_key)
    return _deserialize_payload(body, payload_format)


def _assert_not_leakage_sensitive(meta: dict[str, Any]) -> None:
    stage_name = str(meta.get("stage_name") or "")
    if stage_name in LEAKAGE_SENSITIVE_STAGES or stage_name.startswith("supervised_"):
        raise ValueError(
            f"Refusing to cache leakage-sensitive supervised stage {stage_name!r}. "
            "Fit vectorizers/selectors on TRAIN partitions only."
        )


def has_stage(key: str) -> bool:
    if _l1_get(key) is not None:
        return True
    redis_meta = _redis_get_meta(key)
    if redis_meta is not None:
        return True
    return _meta_path(key).is_file()


def get_stage(key: str) -> dict[str, Any] | None:
    """Load stage sidecar metadata; include ``payload`` when available."""
    cached = _l1_get(key)
    meta: dict[str, Any] | None = None
    if cached is not None and "payload" in cached:
        return cached
    if cached is not None:
        meta = dict(cached)
    else:
        redis_meta = _redis_get_meta(key)
        if redis_meta is not None:
            meta = dict(redis_meta)
        elif _meta_path(key).is_file():
            meta = json.loads(_meta_path(key).read_text(encoding="utf-8"))

    if meta is None:
        return None

    if "payload" not in meta:
        payload_format = meta.get("payload_format", "json")
        try:
            if meta.get("storage_backend") == "object" and meta.get("object_key"):
                meta["payload"] = _load_payload_from_object(meta["object_key"], payload_format)
            else:
                payload_path = meta.get("payload_path")
                if payload_path:
                    resolved = Path(payload_path)
                    if resolved.is_file():
                        meta["payload"] = _load_payload_from_disk(resolved, payload_format)
        except Exception:
            logger.debug("stage cache payload hydrate failed key=%s", key, exc_info=True)

    _l1_put(key, meta)
    return meta


def _persist_object_payload(
    key: str, blob: bytes, payload_format: str
) -> dict[str, str] | None:
    """Upload payload bytes to shared object storage when configured."""
    if not _object_storage_configured():
        return None
    try:
        from backend.core.config import settings
        from backend.core.storage import object_storage

        object_key = f"research-stage-cache/{key}/payload{_payload_extension(payload_format)}"
        object_storage.upload_bytes_sync(
            object_key=object_key,
            body=blob,
            content_type="application/octet-stream",
            metadata={"cache_key": key[:64], "format": payload_format},
        )
        return {
            "storage_backend": "object",
            "object_key": object_key,
            "object_uri": f"s3://{settings.STORAGE_BUCKET}/{object_key}",
        }
    except Exception:
        logger.warning(
            "stage cache object upload failed; keeping local L3 only key=%s",
            key,
            exc_info=True,
        )
        return None


def put_stage(
    key: str,
    *,
    meta: dict[str, Any],
    payload: Any | None = None,
    payload_format: PayloadFormat = "json",
) -> str:
    """Persist stage metadata (and optional payload); returns artifact directory path."""
    _assert_not_leakage_sensitive(meta)
    if has_stage(key):
        return str(_entry_dir(key))

    entry = _entry_dir(key)
    entry.mkdir(parents=True, exist_ok=True)

    sidecar = dict(meta)
    sidecar["cache_key"] = key
    sidecar.setdefault("created_at", datetime.now(UTC).isoformat())
    sidecar.setdefault("cache_layers", ["l1", "l2", "l3"])

    artifact_path = str(entry)
    object_key: str | None = None

    if payload is not None:
        sidecar["payload_format"] = payload_format
        blob = _serialize_payload(payload, payload_format)
        payload_path = entry / f"payload{_payload_extension(payload_format)}"
        payload_path.write_bytes(blob)
        sidecar["payload_path"] = str(payload_path)
        sidecar["storage_backend"] = "local"

        object_meta = _persist_object_payload(key, blob, payload_format)
        if object_meta:
            sidecar.update(object_meta)
            object_key = object_meta.get("object_key")

    _meta_path(key).write_text(
        json.dumps(sidecar, sort_keys=True, ensure_ascii=True, default=str),
        encoding="utf-8",
    )

    publish = dict(sidecar)
    if payload is not None:
        publish["payload"] = payload
    _redis_set_meta(key, sidecar)
    _l1_put(key, publish)

    artifact_registry.register(
        "stage",
        key,
        {
            "artifact_path": artifact_path,
            "stage_name": sidecar.get("stage_name"),
            "payload_format": sidecar.get("payload_format"),
            "storage_backend": sidecar.get("storage_backend"),
            "object_key": object_key,
        },
        key,
    )
    return artifact_path


def invalidate(key: str | None = None) -> None:
    """Remove one cached stage or clear the entire stage cache (all layers)."""
    if key is None:
        _l1_delete(None)
        root = _stage_cache_root()
        if root.exists():
            shutil.rmtree(root)
        return

    _l1_delete(key)
    _redis_delete(key)
    entry = _entry_dir(key)
    object_key = None
    meta_file = _meta_path(key)
    if meta_file.is_file():
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            object_key = meta.get("object_key")
        except Exception:
            object_key = None
    if entry.exists():
        shutil.rmtree(entry)
    if object_key:
        try:
            from backend.core.storage import object_storage

            object_storage.delete_object_sync(object_key)
        except Exception:
            logger.debug("stage cache object delete failed key=%s", key, exc_info=True)


def remember_computation(
    *,
    spec_hash: str,
    corpus_snapshot_hash: str,
    engine_version: str,
    meta: dict[str, Any],
    payload: Any | None = None,
    payload_format: PayloadFormat = "json",
) -> str:
    """Store a full-analysis result under its computation identity key."""
    key = compute_identity_lookup(spec_hash, corpus_snapshot_hash, engine_version)
    sidecar = dict(meta)
    sidecar.setdefault("stage_name", "computation")
    sidecar["spec_hash"] = spec_hash
    sidecar["corpus_snapshot_hash"] = corpus_snapshot_hash
    sidecar["engine_version"] = engine_version
    return put_stage(key, meta=sidecar, payload=payload, payload_format=payload_format)


def get_or_compute(
    key: str,
    factory: Callable[[], tuple[dict[str, Any], Any | None]],
    *,
    payload_format: PayloadFormat = "json",
) -> dict[str, Any]:
    """Stampede-safe cache fill: lookup → lock → re-lookup → compute → publish.

    ``factory`` returns ``(meta, payload)``. Supervised leakage-sensitive stage
    names in ``meta`` are rejected by :func:`put_stage`.
    """
    hit = get_stage(key)
    if hit is not None:
        return hit

    with distributed_lock(key) as held:
        hit = get_stage(key)
        if hit is not None:
            return hit

        if not held:
            deadline = time.monotonic() + min(5.0, _lock_wait())
            while time.monotonic() < deadline:
                hit = get_stage(key)
                if hit is not None:
                    return hit
                if _redis_get_status(key) == "ready":
                    hit = get_stage(key)
                    if hit is not None:
                        return hit
                time.sleep(0.05)

        _redis_set_status(key, "computing")
        try:
            meta, payload = factory()
            put_stage(key, meta=meta, payload=payload, payload_format=payload_format)
            loaded = get_stage(key)
            if loaded is None:
                raise RuntimeError(f"stage cache put failed to materialize key={key}")
            return loaded
        except Exception:
            _redis_set_status(key, "failed")
            raise


def cache_layer_metrics() -> dict[str, Any]:
    """Lightweight observability for L1 occupancy (tests / ops)."""
    with _L1_LOCK:
        return {
            "l1_entries": len(_L1),
            "l1_bytes": _L1_BYTES,
            "redis_enabled": _redis_enabled() and _get_sync_redis() is not None,
            "object_storage": _object_storage_configured(),
        }
