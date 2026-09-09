"""User-defined dictionary matching for research text analysis.

Dictionaries are never shipped as substantive defaults — callers always supply
the definition (flat term list or hierarchical category tree).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

_WORD_RE = re.compile(r"\S+")
_NON_WORD_RE = re.compile(r"[^\w']", re.UNICODE)

MATCH_TYPES: frozenset[str] = frozenset({"token", "phrase", "wildcard", "regex"})
FORMAT_FLAT = "flat"
FORMAT_HIERARCHICAL = "hierarchical_v1"


@dataclass(frozen=True, slots=True)
class DictionaryEntry:
    expression: str
    match_type: str
    category: str | None = None
    subcategory: str | None = None
    path: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "expression": self.expression,
            "match_type": self.match_type,
            "category": self.category,
            "subcategory": self.subcategory,
            "path": list(self.path),
        }


@dataclass
class DictionarySpec:
    """Normalized, matchable dictionary definition (always user-supplied)."""

    entries: list[DictionaryEntry] = field(default_factory=list)
    exclusions: list[DictionaryEntry] = field(default_factory=list)
    name: str | None = None
    version: str | None = None
    description: str | None = None
    language: str | None = None
    format: str = FORMAT_FLAT
    source: str = "user"  # never "builtin_research"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "language": self.language,
            "format": self.format,
            "source": self.source,
            "entry_count": len(self.entries),
            "exclusion_count": len(self.exclusions),
            "entries": [e.to_dict() for e in self.entries],
            "exclusions": [e.to_dict() for e in self.exclusions],
        }

    def flattened_terms(self) -> list[str]:
        """Leaf expressions for APIs that still expect a flat term list."""
        seen: set[str] = set()
        out: list[str] = []
        for entry in self.entries:
            key = entry.expression
            if key not in seen:
                seen.add(key)
                out.append(key)
        return out


def infer_match_type(expression: str, explicit: str | None = None) -> str:
    if explicit:
        key = str(explicit).strip().lower().replace("-", "_")
        aliases = {
            "exact": "token",
            "exact_token": "token",
            "word": "token",
            "exact_phrase": "phrase",
            "mwe": "phrase",
            "multiword": "phrase",
            "multi_word": "phrase",
            "glob": "wildcard",
        }
        key = aliases.get(key, key)
        if key not in MATCH_TYPES:
            raise ValueError(
                f"Unsupported dictionary match type {explicit!r}; "
                f"expected one of {', '.join(sorted(MATCH_TYPES))}"
            )
        return key
    text = expression.strip()
    if any(ch in text for ch in "*?"):
        return "wildcard"
    if " " in text:
        return "phrase"
    return "token"


def _fold(value: str, *, case_sensitive: bool) -> str:
    return value if case_sensitive else value.casefold()


def _token_key(token: str, *, case_sensitive: bool) -> str:
    return _fold(_NON_WORD_RE.sub("", token), case_sensitive=case_sensitive)


def _wildcard_regex(pattern: str, *, case_sensitive: bool) -> re.Pattern[str]:
    parts: list[str] = []
    for ch in pattern:
        if ch == "*":
            parts.append(".*")
        elif ch == "?":
            parts.append(".")
        else:
            parts.append(re.escape(ch))
    flags = 0 if case_sensitive else re.IGNORECASE
    return re.compile("^" + "".join(parts) + "$", flags)


def _parse_leaf(item: Any, path: tuple[str, ...]) -> DictionaryEntry:
    category = path[0] if path else None
    subcategory = "/".join(path[1:]) if len(path) >= 2 else None

    if isinstance(item, str):
        expr = item.strip()
        if not expr:
            raise ValueError("Dictionary expression must be non-empty")
        return DictionaryEntry(
            expression=expr,
            match_type=infer_match_type(expr),
            category=category,
            subcategory=subcategory,
            path=path,
        )
    if isinstance(item, dict):
        expr = str(item.get("expression") or item.get("term") or item.get("phrase") or "").strip()
        if not expr:
            raise ValueError(f"Dictionary entry missing expression: {item!r}")
        match_type = infer_match_type(expr, item.get("match") or item.get("match_type"))
        return DictionaryEntry(
            expression=expr,
            match_type=match_type,
            category=str(item["category"]) if item.get("category") is not None else category,
            subcategory=(
                str(item["subcategory"]) if item.get("subcategory") is not None else subcategory
            ),
            path=path,
        )
    raise ValueError(f"Unsupported dictionary leaf {item!r}")


def _walk_hierarchy(node: Any, path: tuple[str, ...] = ()) -> Iterator[DictionaryEntry]:
    if isinstance(node, list):
        for item in node:
            yield _parse_leaf(item, path)
        return
    if isinstance(node, dict):
        # Detect envelope keys vs category keys.
        for key, value in node.items():
            yield from _walk_hierarchy(value, path + (str(key),))
        return
    if isinstance(node, str):
        yield _parse_leaf(node, path)
        return
    raise ValueError(f"Unsupported hierarchy node at {'/'.join(path) or '<root>'}: {node!r}")


def _parse_exclusion_list(raw: Any) -> list[DictionaryEntry]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("exclusions must be a list of expressions or entry objects")
    return [_parse_leaf(item, ()) for item in raw]


def parse_dictionary_payload(
    payload: Any,
    *,
    name: str | None = None,
    version: str | None = None,
    description: str | None = None,
    language: str | None = None,
) -> DictionarySpec:
    """Parse flat term lists or hierarchical dict/YAML-shaped JSON into a spec."""
    if payload is None:
        raise ValueError("Dictionary definition is required (user-defined; no built-in defaults)")

    if isinstance(payload, str):
        payload = json.loads(payload)

    if isinstance(payload, list):
        entries = [_parse_leaf(item, ()) for item in payload]
        return DictionarySpec(
            entries=entries,
            exclusions=[],
            name=name,
            version=version,
            description=description,
            language=language,
            format=FORMAT_FLAT,
            source="user",
        )

    if not isinstance(payload, dict):
        raise ValueError("Dictionary payload must be a list of terms or a hierarchical object")

    # Envelope form:
    # { language, exclusions, hierarchy|categories|entries|terms, ... }
    lang = language if language is not None else payload.get("language")
    desc = description if description is not None else payload.get("description")
    ver = version if version is not None else payload.get("version")
    nm = name if name is not None else payload.get("name")
    exclusions = _parse_exclusion_list(payload.get("exclusions"))

    hierarchy = None
    for key in ("hierarchy", "categories", "entries", "tree"):
        if key in payload and payload[key] is not None:
            hierarchy = payload[key]
            break

    if hierarchy is not None:
        entries = list(_walk_hierarchy(hierarchy))
        return DictionarySpec(
            entries=entries,
            exclusions=exclusions,
            name=nm,
            version=ver,
            description=desc,
            language=str(lang) if lang is not None else None,
            format=FORMAT_HIERARCHICAL,
            source="user",
        )

    if "terms" in payload:
        terms = payload["terms"]
        if not isinstance(terms, list):
            raise ValueError("terms must be a list")
        entries = [_parse_leaf(item, ()) for item in terms]
        return DictionarySpec(
            entries=entries,
            exclusions=exclusions,
            name=nm,
            version=ver,
            description=desc,
            language=str(lang) if lang is not None else None,
            format=FORMAT_FLAT,
            source="user",
        )

    # Bare hierarchy object (YAML shape): category -> subcategory -> [terms]
    # Treat whole dict as hierarchy when it does not look like an envelope.
    envelope_keys = {
        "language",
        "exclusions",
        "description",
        "version",
        "name",
        "format",
        "source",
    }
    if payload and all(k in envelope_keys for k in payload):
        raise ValueError("Hierarchical dictionary is missing hierarchy/categories/terms")

    entries = list(_walk_hierarchy(payload))
    return DictionarySpec(
        entries=entries,
        exclusions=exclusions,
        name=nm,
        version=ver,
        description=desc,
        language=str(lang) if lang is not None else None,
        format=FORMAT_HIERARCHICAL,
        source="user",
    )


def serialize_dictionary_payload(
    spec: DictionarySpec, *, hierarchy: dict[str, Any] | None = None
) -> str:
    """Serialize a spec to JSON for ``terms_json`` storage."""
    if hierarchy is not None:
        payload: dict[str, Any] = {
            "format": FORMAT_HIERARCHICAL,
            "source": "user",
            "hierarchy": hierarchy,
            "exclusions": [
                {"expression": e.expression, "match": e.match_type} for e in spec.exclusions
            ],
        }
        if spec.language:
            payload["language"] = spec.language
        if spec.description:
            payload["description"] = spec.description
        if spec.version:
            payload["version"] = spec.version
        if spec.name:
            payload["name"] = spec.name
        return json.dumps(payload, ensure_ascii=True)

    if spec.format == FORMAT_HIERARCHICAL:
        tree: dict[str, Any] = {}
        for entry in spec.entries:
            if not entry.path:
                bucket = tree.setdefault("_terms", [])
                if not isinstance(bucket, list):
                    raise ValueError("Cannot serialize overlapping hierarchy paths")
                bucket.append({"expression": entry.expression, "match": entry.match_type})
                continue
            cursor: Any = tree
            for part in entry.path[:-1]:
                nxt = cursor.setdefault(part, {})
                if not isinstance(nxt, dict):
                    raise ValueError("Cannot serialize overlapping hierarchy paths")
                cursor = nxt
            leaf_key = entry.path[-1]
            leaf = cursor.setdefault(leaf_key, [])
            if not isinstance(leaf, list):
                raise ValueError("Cannot serialize overlapping hierarchy paths")
            leaf.append({"expression": entry.expression, "match": entry.match_type})
        return serialize_dictionary_payload(spec, hierarchy=tree)

    if not spec.exclusions and not spec.language:
        return json.dumps(spec.flattened_terms(), ensure_ascii=True)
    payload = {
        "format": FORMAT_FLAT,
        "source": "user",
        "terms": [{"expression": e.expression, "match": e.match_type} for e in spec.entries],
        "exclusions": [
            {"expression": e.expression, "match": e.match_type} for e in spec.exclusions
        ],
    }
    if spec.language:
        payload["language"] = spec.language
    return json.dumps(payload, ensure_ascii=True)


def hierarchy_from_payload(payload: Any) -> dict[str, Any] | None:
    """Extract hierarchy object for API responses, if present."""
    if payload is None:
        return None
    if isinstance(payload, str):
        payload = json.loads(payload)
    if isinstance(payload, list):
        return None
    if not isinstance(payload, dict):
        return None
    for key in ("hierarchy", "categories", "entries", "tree"):
        if isinstance(payload.get(key), dict):
            return payload[key]
    envelope_keys = {
        "language",
        "exclusions",
        "description",
        "version",
        "name",
        "format",
        "source",
        "terms",
    }
    if "terms" in payload:
        return None
    if payload and not any(k in envelope_keys for k in payload):
        return payload
    # Mixed: only non-envelope keys form hierarchy
    tree = {k: v for k, v in payload.items() if k not in envelope_keys}
    return tree or None


def _match_entry_on_tokens(
    tokens: list[str],
    entry: DictionaryEntry,
    *,
    case_sensitive: bool,
) -> Iterator[tuple[int, int, list[str]]]:
    """Yield (start, end, matched_token_texts) for each hit of ``entry``."""
    n = len(tokens)
    if entry.match_type == "token":
        needle = _token_key(entry.expression, case_sensitive=case_sensitive)
        if not needle:
            return
        for i, tok in enumerate(tokens):
            if _token_key(tok, case_sensitive=case_sensitive) == needle:
                yield i, i + 1, [tok]
        return

    if entry.match_type == "phrase":
        parts = [p for p in _WORD_RE.findall(entry.expression.strip()) if p]
        if not parts:
            return
        keys = [_token_key(p, case_sensitive=case_sensitive) for p in parts]
        width = len(keys)
        for start in range(0, n - width + 1):
            window = tokens[start : start + width]
            if all(
                _token_key(tok, case_sensitive=case_sensitive) == key
                for tok, key in zip(window, keys, strict=True)
            ):
                yield start, start + width, list(window)
        return

    if entry.match_type == "wildcard":
        parts = [p for p in _WORD_RE.findall(entry.expression.strip()) if p]
        if not parts:
            return
        patterns = [_wildcard_regex(p, case_sensitive=case_sensitive) for p in parts]
        width = len(patterns)
        for start in range(0, n - width + 1):
            window = tokens[start : start + width]
            if all(
                pat.match(_NON_WORD_RE.sub("", tok) or tok)
                for tok, pat in zip(window, patterns, strict=True)
            ):
                yield start, start + width, list(window)
        return

    if entry.match_type == "regex":
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            compiled = re.compile(entry.expression, flags)
        except re.error as exc:
            raise ValueError(f"Invalid dictionary regex {entry.expression!r}: {exc}") from exc
        for i, tok in enumerate(tokens):
            if compiled.search(tok):
                yield i, i + 1, [tok]
        joined = " ".join(tokens)
        offsets: list[tuple[int, int, int]] = []
        pos = 0
        for i, tok in enumerate(tokens):
            if i:
                pos += 1
            start = pos
            end = pos + len(tok)
            offsets.append((start, end, i))
            pos = end
        for match in compiled.finditer(joined):
            m0, m1 = match.start(), match.end()
            if m0 == m1:
                continue
            first = last = None
            for start, end, idx in offsets:
                if end <= m0 or start >= m1:
                    continue
                if first is None:
                    first = idx
                last = idx
            if first is None or last is None or first == last:
                continue
            yield first, last + 1, tokens[first : last + 1]
        return

    raise ValueError(f"Unsupported match type {entry.match_type!r}")


def _spans_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return a0 < b1 and b0 < a1


def match_dictionary(
    tokenized: list[list[str]],
    spec: DictionarySpec,
    *,
    unit_ids: list[str] | None = None,
    metadata: list[dict[str, Any]] | None = None,
    case_sensitive: bool = False,
    rate_per: float = 1000.0,
) -> dict[str, Any]:
    """Apply a user-defined dictionary to tokenized units.

    Returns totals, normalized rates, prevalence, per-unit summaries, and
    individual match records with category / expression / provenance.
    """
    if rate_per <= 0:
        raise ValueError("rate_per must be > 0")
    n_units = len(tokenized)
    if unit_ids is not None and len(unit_ids) != n_units:
        raise ValueError("unit_ids must align 1:1 with tokenized units")
    if metadata is not None and len(metadata) != n_units:
        raise ValueError("metadata must align 1:1 with tokenized units")
    if not spec.entries:
        raise ValueError("Dictionary has no entries (user-defined dictionaries cannot be empty)")

    total_tokens = sum(len(toks) for toks in tokenized)
    matches: list[dict[str, Any]] = []
    per_unit_hits = [0 for _ in range(n_units)]
    by_category: dict[str, dict[str, Any]] = {}

    for unit_index, tokens in enumerate(tokenized):
        meta = dict(metadata[unit_index]) if metadata else {}
        unit_id = (
            unit_ids[unit_index]
            if unit_ids is not None
            else meta.get("text_unit_id", str(unit_index))
        )

        exclusion_spans: list[tuple[int, int]] = []
        for excl in spec.exclusions:
            for start, end, _ in _match_entry_on_tokens(
                tokens, excl, case_sensitive=case_sensitive
            ):
                exclusion_spans.append((start, end))

        for entry in spec.entries:
            for start, end, matched_tokens in _match_entry_on_tokens(
                tokens, entry, case_sensitive=case_sensitive
            ):
                if any(_spans_overlap(start, end, e0, e1) for e0, e1 in exclusion_spans):
                    continue
                per_unit_hits[unit_index] += 1
                cat_key = entry.category or "_uncategorized"
                bucket = by_category.setdefault(
                    cat_key,
                    {"category": entry.category, "hits": 0, "subcategories": {}},
                )
                bucket["hits"] = int(bucket["hits"]) + 1
                if entry.subcategory:
                    sub = bucket["subcategories"].setdefault(
                        entry.subcategory, {"hits": 0, "expressions": {}}
                    )
                    sub["hits"] = int(sub["hits"]) + 1
                    expr_bucket = sub["expressions"].setdefault(entry.expression, 0)
                    sub["expressions"][entry.expression] = int(expr_bucket) + 1

                hit = {
                    "text_unit_id": unit_id,
                    "matched_expression": entry.expression,
                    "match_type": entry.match_type,
                    "matched_tokens": matched_tokens,
                    "token_start": start,
                    "token_end": end,
                    "category": entry.category,
                    "subcategory": entry.subcategory,
                    "path": list(entry.path),
                }
                for key, value in meta.items():
                    if key == "text":
                        continue
                    hit.setdefault(key, value)
                matches.append(hit)

    total_hits = sum(per_unit_hits)
    units_with_hit = sum(1 for h in per_unit_hits if h > 0)
    normalized = (total_hits / total_tokens * rate_per) if total_tokens else 0.0
    prevalence = (units_with_hit / n_units) if n_units else 0.0

    per_unit = []
    for i, hits in enumerate(per_unit_hits):
        row: dict[str, Any] = {
            "text_unit_id": unit_ids[i] if unit_ids is not None else str(i),
            "hits": hits,
        }
        if metadata:
            for key, value in metadata[i].items():
                if key != "text":
                    row.setdefault(key, value)
        per_unit.append(row)

    return {
        "total_hits": total_hits,
        "normalized_hits": normalized,
        "hits_per_1000_tokens": (total_hits / total_tokens * 1000.0) if total_tokens else 0.0,
        "rate_per": rate_per,
        "document_prevalence": prevalence,
        "unit_prevalence": prevalence,
        "n_units": n_units,
        "n_tokens": total_tokens,
        "units_with_hit": units_with_hit,
        "per_unit": per_unit,
        "matches": matches,
        "by_category": by_category,
        "dictionary": spec.to_dict(),
        "case_sensitive": case_sensitive,
    }


# Synthetic fixture helper — clearly marked, not a research default.
SYNTHETIC_DEMO_DICTIONARY: dict[str, Any] = {
    "format": FORMAT_HIERARCHICAL,
    "source": "synthetic_demo",
    "description": "SYNTHETIC demo fixture only — not a substantive research dictionary.",
    "language": "en",
    "hierarchy": {
        "demo_category": {
            "demo_subcategory": [
                "alpha",
                "beta gamma",
                {"expression": "delta*", "match": "wildcard"},
            ]
        }
    },
    "exclusions": ["beta gamma noise"],
}
