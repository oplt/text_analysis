"""Duplicate / near-duplicate detection for research corpora.

Four tiers, cheapest first:

* ``exact`` — sha256 checksum of the raw text
* ``normalized`` — sha256 checksum after whitespace/case normalization
* ``lexical`` — character n-gram Jaccard overlap (the same metric ingestion
  QA already used for its near-duplicate heuristic)
* ``minhash`` — optional pure-Python MinHash + LSH banding, for corpora where
  full pairwise n-gram comparison would not scale. ``datasketch`` is not a
  dependency of this project (see ``backend/pyproject.toml``), so a small,
  seeded, dependency-free MinHash estimator is implemented directly here.

These are pure functions over ``{"id": ..., "text": ...}`` items — nothing
here reaches into the database or FastAPI layer, and nothing assumes a
specific document type, research question, or corpus size.

This module is the single implementation used both as an explicit
"duplicate detection" analysis and by ingestion QA
(:mod:`backend.modules.text_research.infrastructure.ingestion_qa`), so the
two call sites can never silently disagree on what counts as a duplicate.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from backend.modules.text_research.infrastructure.canonical_text import sha256_text

DUPLICATE_METHODS: frozenset[str] = frozenset({"exact", "normalized", "lexical", "minhash"})

DEFAULT_LEXICAL_THRESHOLD = 0.85
DEFAULT_CHAR_NGRAM_SIZE = 5
DEFAULT_MINHASH_NUM_PERM = 64
DEFAULT_MINHASH_SHINGLE_SIZE = 3
DEFAULT_MINHASH_THRESHOLD = 0.8
DEFAULT_MINHASH_BANDS = 16
DEFAULT_MAX_PAIRS = 1000

_WHITESPACE_RE = re.compile(r"\s+")

# 61-bit Mersenne prime — standard modulus for MinHash universal hashing.
_MERSENNE_PRIME = (1 << 61) - 1
_MAX_HASH_32 = (1 << 32) - 1


def normalize_duplicate_methods(methods: Sequence[str] | None) -> list[str]:
    resolved = list(methods) if methods else ["exact", "normalized", "lexical"]
    unknown = sorted(set(resolved) - DUPLICATE_METHODS)
    if unknown:
        raise ValueError(
            f"Unsupported duplicate detection method(s) {unknown}; "
            f"expected any of {sorted(DUPLICATE_METHODS)}"
        )
    return resolved


def normalize_text_for_checksum(text: str) -> str:
    """Collapse whitespace and lowercase — used for the ``normalized`` tier."""
    return _WHITESPACE_RE.sub(" ", (text or "").strip()).lower()


def exact_checksum(text: str) -> str:
    """sha256 over the raw text, unmodified."""
    return sha256_text(text or "")


def normalized_checksum(text: str) -> str:
    """sha256 over whitespace/case-normalized text."""
    return sha256_text(normalize_text_for_checksum(text))


def char_ngrams(text: str, n: int = DEFAULT_CHAR_NGRAM_SIZE) -> Counter[str]:
    """Character n-gram multiset of the whitespace/case-normalized text."""
    if n < 1:
        raise ValueError("n must be >= 1")
    normalized = normalize_text_for_checksum(text)
    if len(normalized) < n:
        return Counter([normalized] if normalized else [])
    return Counter(normalized[i : i + n] for i in range(len(normalized) - n + 1))


def jaccard_counters(a: Counter[str], b: Counter[str]) -> float:
    """Multiset (weighted) Jaccard overlap between two n-gram counters."""
    if not a and not b:
        return 1.0
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    intersection = sum(min(a[k], b[k]) for k in keys)
    union = sum(max(a[k], b[k]) for k in keys)
    return intersection / union if union else 0.0


def lexical_similarity(text_a: str, text_b: str, *, n: int = DEFAULT_CHAR_NGRAM_SIZE) -> float:
    """Character n-gram Jaccard similarity between two raw texts."""
    return jaccard_counters(char_ngrams(text_a, n), char_ngrams(text_b, n))


def group_by_key(rows: Sequence[tuple[str, str]]) -> list[dict[str, Any]]:
    """``rows``: ``[(id, key), ...]`` -> groups of ids sharing a key (size >= 2)."""
    grouped: dict[str, list[str]] = defaultdict(list)
    for item_id, key in rows:
        grouped[key].append(item_id)
    return [
        {"key": key, "ids": ids, "count": len(ids)} for key, ids in grouped.items() if len(ids) >= 2
    ]


# ---------------------------------------------------------------------------
# MinHash (pure Python; datasketch is not a project dependency)
# ---------------------------------------------------------------------------


def _stable_hash32(token: str) -> int:
    """Deterministic 32-bit hash, independent of ``PYTHONHASHSEED``."""
    return int(hashlib.md5(token.encode("utf-8")).hexdigest()[:8], 16)


def _word_shingles(text: str, k: int) -> set[str]:
    normalized = normalize_text_for_checksum(text)
    tokens = normalized.split()
    if not tokens:
        return set()
    if len(tokens) < k:
        return {" ".join(tokens)}
    return {" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}


def _minhash_permutations(num_perm: int, seed: int) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    return [
        (rng.randrange(1, _MERSENNE_PRIME), rng.randrange(0, _MERSENNE_PRIME))
        for _ in range(num_perm)
    ]


@dataclass(frozen=True, slots=True)
class MinHashSignature:
    """A fixed-length MinHash signature — an unbiased Jaccard-similarity estimator."""

    values: tuple[int, ...]
    num_perm: int
    shingle_size: int

    def to_dict(self) -> dict[str, Any]:
        return {"num_perm": self.num_perm, "shingle_size": self.shingle_size}


def minhash_signature(
    text: str,
    *,
    num_perm: int = DEFAULT_MINHASH_NUM_PERM,
    shingle_size: int = DEFAULT_MINHASH_SHINGLE_SIZE,
    seed: int = 1,
) -> MinHashSignature:
    """Compute a MinHash signature over word shingles of ``text``.

    Implemented in pure Python (no ``datasketch`` dependency). Deterministic
    given the same ``seed``/``num_perm``/``shingle_size`` — safe to persist
    and reproduce.
    """
    if num_perm < 1:
        raise ValueError("num_perm must be >= 1")
    if shingle_size < 1:
        raise ValueError("shingle_size must be >= 1")

    shingles = _word_shingles(text, shingle_size)
    if not shingles:
        return MinHashSignature(
            values=tuple([_MAX_HASH_32] * num_perm), num_perm=num_perm, shingle_size=shingle_size
        )

    hashed = [_stable_hash32(s) for s in shingles]
    permutations = _minhash_permutations(num_perm, seed)
    signature = tuple(
        min(((a * h + b) % _MERSENNE_PRIME) & _MAX_HASH_32 for h in hashed) for a, b in permutations
    )
    return MinHashSignature(values=signature, num_perm=num_perm, shingle_size=shingle_size)


def minhash_jaccard(sig_a: MinHashSignature, sig_b: MinHashSignature) -> float:
    """Estimated Jaccard similarity from two comparable MinHash signatures."""
    if sig_a.num_perm != sig_b.num_perm:
        raise ValueError("MinHash signatures must share the same num_perm to compare")
    if sig_a.num_perm == 0:
        return 0.0
    matches = sum(1 for x, y in zip(sig_a.values, sig_b.values, strict=True) if x == y)
    return matches / sig_a.num_perm


def minhash_lsh_candidate_pairs(
    signatures: dict[str, MinHashSignature],
    *,
    num_bands: int | None = None,
) -> set[tuple[str, str]]:
    """LSH banding: return id pairs that share at least one band bucket.

    A cheap O(n) pre-filter for large corpora that avoids exhaustive O(n²)
    MinHash comparisons. Candidate pairs still get a full similarity check
    before being reported — banding never fabricates a score.
    """
    ids = list(signatures)
    if len(ids) < 2:
        return set()
    num_perm = next(iter(signatures.values())).num_perm
    bands = max(1, min(num_bands or DEFAULT_MINHASH_BANDS, num_perm))
    rows_per_band = max(1, num_perm // bands)

    buckets: dict[tuple[int, tuple[int, ...]], list[str]] = defaultdict(list)
    for doc_id in ids:
        sig = signatures[doc_id].values
        for band in range(bands):
            start = band * rows_per_band
            end = start + rows_per_band
            buckets[(band, tuple(sig[start:end]))].append(doc_id)

    candidates: set[tuple[str, str]] = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                candidates.add(tuple(sorted((members[i], members[j]))))  # type: ignore[arg-type]
    return candidates


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def duplicate_report(
    items: Sequence[dict[str, Any]],
    *,
    methods: Sequence[str] | None = None,
    lexical_threshold: float = DEFAULT_LEXICAL_THRESHOLD,
    char_ngram_size: int = DEFAULT_CHAR_NGRAM_SIZE,
    minhash_num_perm: int = DEFAULT_MINHASH_NUM_PERM,
    minhash_shingle_size: int = DEFAULT_MINHASH_SHINGLE_SIZE,
    minhash_threshold: float = DEFAULT_MINHASH_THRESHOLD,
    minhash_use_lsh: bool = True,
    minhash_bands: int = DEFAULT_MINHASH_BANDS,
    max_pairs: int | None = DEFAULT_MAX_PAIRS,
    force_lexical: bool = False,
) -> dict[str, Any]:
    """Run the requested duplicate-detection tiers over ``items``.

    ``items``: ``[{"id": str, "text": str}, ...]``. Every tier is
    independently reported; nothing here assumes a specific document type,
    research question, or corpus size.

    On large corpora, the O(n²) lexical tier is skipped unless
    ``force_lexical=True``; minhash is auto-enabled when needed.
    """
    from backend.modules.text_research.infrastructure.out_of_core import (
        duplicate_methods_for_scale,
    )

    resolved_methods = normalize_duplicate_methods(methods)
    resolved_methods, scale_notes = duplicate_methods_for_scale(
        resolved_methods, len([i for i in items if i.get("id")]), force_lexical=force_lexical
    )
    if not 0.0 <= lexical_threshold <= 1.0:
        raise ValueError("lexical_threshold must be between 0 and 1")
    if not 0.0 <= minhash_threshold <= 1.0:
        raise ValueError("minhash_threshold must be between 0 and 1")
    if char_ngram_size < 1:
        raise ValueError("char_ngram_size must be >= 1")

    prepared = [
        {"id": item["id"], "text": item.get("text") or ""} for item in items if item.get("id")
    ]
    n = len(prepared)

    result: dict[str, Any] = {
        "methods": resolved_methods,
        "item_count": n,
        "exact_duplicate_groups": [],
        "normalized_duplicate_groups": [],
        "lexical_near_duplicates": [],
        "minhash_near_duplicates": [],
        "out_of_core_notes": scale_notes,
    }

    if "exact" in resolved_methods:
        rows = [(row["id"], exact_checksum(row["text"])) for row in prepared]
        result["exact_duplicate_groups"] = group_by_key(rows)

    if "normalized" in resolved_methods:
        rows = [(row["id"], normalized_checksum(row["text"])) for row in prepared]
        result["normalized_duplicate_groups"] = group_by_key(rows)

    if "lexical" in resolved_methods:
        grams = [char_ngrams(row["text"], char_ngram_size) for row in prepared]
        pairs: list[dict[str, Any]] = []
        for i in range(n):
            if not prepared[i]["text"].strip():
                continue
            for j in range(i + 1, n):
                if not prepared[j]["text"].strip():
                    continue
                score = jaccard_counters(grams[i], grams[j])
                if score >= lexical_threshold:
                    pairs.append(
                        {
                            "source_id": prepared[i]["id"],
                            "target_id": prepared[j]["id"],
                            "score": round(score, 4),
                            "method": "lexical_char_ngram",
                            "char_ngram_size": char_ngram_size,
                        }
                    )
        pairs.sort(key=lambda row: -row["score"])
        result["lexical_near_duplicates"] = pairs[:max_pairs] if max_pairs else pairs

    if "minhash" in resolved_methods:
        signatures = {
            row["id"]: minhash_signature(
                row["text"], num_perm=minhash_num_perm, shingle_size=minhash_shingle_size
            )
            for row in prepared
            if row["text"].strip()
        }
        use_lsh = minhash_use_lsh and len(signatures) > 2
        if use_lsh:
            candidate_pairs = minhash_lsh_candidate_pairs(signatures, num_bands=minhash_bands)
        else:
            sig_ids = list(signatures)
            candidate_pairs = {
                (sig_ids[i], sig_ids[j])
                for i in range(len(sig_ids))
                for j in range(i + 1, len(sig_ids))
            }
        pairs = []
        for a, b in candidate_pairs:
            score = minhash_jaccard(signatures[a], signatures[b])
            if score >= minhash_threshold:
                pairs.append(
                    {
                        "source_id": a,
                        "target_id": b,
                        "score": round(score, 4),
                        "method": "minhash",
                        "num_perm": minhash_num_perm,
                        "shingle_size": minhash_shingle_size,
                    }
                )
        pairs.sort(key=lambda row: -row["score"])
        result["minhash_near_duplicates"] = pairs[:max_pairs] if max_pairs else pairs
        result["minhash_used_lsh"] = use_lsh
        result["minhash_candidate_pairs_checked"] = len(candidate_pairs)

    result["summary"] = {
        "exact_duplicate_groups": len(result["exact_duplicate_groups"]),
        "normalized_duplicate_groups": len(result["normalized_duplicate_groups"]),
        "lexical_near_duplicate_pairs": len(result["lexical_near_duplicates"]),
        "minhash_near_duplicate_pairs": len(result["minhash_near_duplicates"]),
    }
    return result


def describe_duplicate_detection_capabilities() -> dict[str, Any]:
    return {
        "methods": sorted(DUPLICATE_METHODS),
        "tiers": {
            "exact": "sha256 checksum of the raw text",
            "normalized": "sha256 checksum after whitespace/case normalization",
            "lexical": "character n-gram Jaccard overlap (pairwise; O(n^2))",
            "minhash": "pure-Python MinHash + LSH banding (optional; scales to larger corpora)",
        },
        "usable_from": ["explicit analysis run", "ingestion QA"],
        "notes": [
            "minhash requires no external dependency (datasketch is not in "
            "pyproject.toml); an unbiased Jaccard estimator is implemented directly "
            "with seeded universal hashing permutations.",
            "LSH banding is a candidate-pair pre-filter, not a similarity guarantee — "
            "reported scores are always the true MinHash-estimated Jaccard for the pair.",
        ],
    }
