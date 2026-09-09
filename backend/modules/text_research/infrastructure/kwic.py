"""Keyword-in-context (KWIC) / concordance search.

Tokenization for context windows is independent of research preprocessing profiles:
matches operate on whitespace tokens with character spans so provenance stays
aligned to the unit text the researcher sees.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterator

_WORD_RE = re.compile(r"\S+")
_NON_WORD_RE = re.compile(r"[^\w']", re.UNICODE)

QUERY_MODES: frozenset[str] = frozenset(
    {
        "auto",
        "word",
        "phrase",
        "exact_phrase",
        "regex",
        "wildcard",
        "lemma",
    }
)

# Token attributes currently supported when the tokenizer/lemmatizer can provide them.
SUPPORTED_TOKEN_ATTRIBUTES: frozenset[str] = frozenset({"surface", "lemma"})


@dataclass(frozen=True, slots=True)
class TokenSpan:
    text: str
    start: int
    end: int

    @property
    def surface(self) -> str:
        return self.text


def tokenize_with_spans(text: str) -> list[TokenSpan]:
    """Split ``text`` into whitespace tokens with character offsets."""
    return [TokenSpan(m.group(0), m.start(), m.end()) for m in _WORD_RE.finditer(text or "")]


def normalize_query_mode(mode: str | None, query: str) -> str:
    """Resolve ``auto`` / aliases into a concrete query mode."""
    raw = (mode or "auto").strip().lower().replace("-", "_")
    aliases = {
        "single": "word",
        "token": "word",
        "exact": "exact_phrase",
        "regexp": "regex",
        "re": "regex",
        "glob": "wildcard",
    }
    key = aliases.get(raw, raw)
    if key not in QUERY_MODES:
        raise ValueError(
            f"Unsupported KWIC query_mode {mode!r}; expected one of "
            f"{', '.join(sorted(QUERY_MODES - {'auto'}))}, or auto"
        )
    if key != "auto":
        return key

    q = query.strip()
    if any(ch in q for ch in "*?"):
        return "wildcard"
    if " " in q:
        return "phrase"
    return "word"


def _fold(value: str, *, case_sensitive: bool) -> str:
    return value if case_sensitive else value.casefold()


def _strip_punct(token: str) -> str:
    return _NON_WORD_RE.sub("", token)


def _token_key(token: str, *, case_sensitive: bool, strip_punct: bool = True) -> str:
    core = _strip_punct(token) if strip_punct else token
    return _fold(core, case_sensitive=case_sensitive)


def _split_query_tokens(query: str) -> list[str]:
    return [t for t in _WORD_RE.findall(query.strip()) if t]


def wildcard_to_regex(pattern: str, *, case_sensitive: bool) -> re.Pattern[str]:
    """Convert a simple ``*`` / ``?`` wildcard pattern to a compiled regex."""
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


def _lemma_of(token: str, language: str | None) -> str:
    from backend.modules.text_research.infrastructure.preprocessing import lemmatize_token
    from backend.modules.text_research.infrastructure.language_processing import (
        lemmatization_available,
    )

    if not language or not lemmatization_available(language):
        raise ValueError(
            "Lemma KWIC requires an available lemmatizer for the requested language; "
            "pass language=… when query_mode='lemma' or token_attribute='lemma'."
        )
    return lemmatize_token(_strip_punct(token) or token, language)


def _iter_word_matches(
    tokens: list[TokenSpan],
    query: str,
    *,
    case_sensitive: bool,
) -> Iterator[tuple[int, int]]:
    needle = _token_key(query, case_sensitive=case_sensitive)
    if not needle:
        return
    for idx, tok in enumerate(tokens):
        if _token_key(tok.text, case_sensitive=case_sensitive) == needle:
            yield idx, idx + 1


def _iter_phrase_matches(
    tokens: list[TokenSpan],
    query: str,
    *,
    case_sensitive: bool,
    strip_punct: bool = True,
) -> Iterator[tuple[int, int]]:
    parts = _split_query_tokens(query)
    if not parts:
        return
    keys = [_token_key(p, case_sensitive=case_sensitive, strip_punct=strip_punct) for p in parts]
    n = len(keys)
    for start in range(0, len(tokens) - n + 1):
        window = tokens[start : start + n]
        if all(
            _token_key(tok.text, case_sensitive=case_sensitive, strip_punct=strip_punct) == key
            for tok, key in zip(window, keys, strict=True)
        ):
            yield start, start + n


def _iter_exact_phrase_char_matches(
    text: str,
    query: str,
    *,
    case_sensitive: bool,
) -> Iterator[tuple[int, int]]:
    """Character-span matches for an exact phrase substring."""
    needle = query.strip()
    if not needle:
        return
    hay = text if case_sensitive else text.casefold()
    needle_cmp = needle if case_sensitive else needle.casefold()
    start = 0
    while True:
        idx = hay.find(needle_cmp, start)
        if idx < 0:
            break
        yield idx, idx + len(needle)
        start = idx + max(len(needle), 1)


def _char_span_to_token_span(
    tokens: list[TokenSpan], char_start: int, char_end: int
) -> tuple[int, int] | None:
    if not tokens:
        return None
    first: int | None = None
    last: int | None = None
    for i, tok in enumerate(tokens):
        if tok.end <= char_start or tok.start >= char_end:
            continue
        if first is None:
            first = i
        last = i
    if first is None or last is None:
        # Fall back to nearest token by start.
        for i, tok in enumerate(tokens):
            if tok.start >= char_start:
                return i, i + 1
        return len(tokens) - 1, len(tokens)
    return first, last + 1


def _iter_regex_matches(
    text: str,
    pattern: str,
    *,
    case_sensitive: bool,
) -> Iterator[tuple[int, int]]:
    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        compiled = re.compile(pattern, flags)
    except re.error as exc:
        raise ValueError(f"Invalid KWIC regex: {exc}") from exc
    for match in compiled.finditer(text):
        if match.end() == match.start():
            continue
        yield match.start(), match.end()


def _iter_wildcard_matches(
    tokens: list[TokenSpan],
    query: str,
    *,
    case_sensitive: bool,
) -> Iterator[tuple[int, int]]:
    parts = _split_query_tokens(query)
    if not parts:
        return
    patterns = [wildcard_to_regex(p, case_sensitive=case_sensitive) for p in parts]
    n = len(patterns)
    for start in range(0, len(tokens) - n + 1):
        window = tokens[start : start + n]
        if all(pat.match(_strip_punct(tok.text) or tok.text) for tok, pat in zip(window, patterns, strict=True)):
            yield start, start + n


def _iter_lemma_matches(
    tokens: list[TokenSpan],
    query: str,
    *,
    case_sensitive: bool,
    language: str | None,
) -> Iterator[tuple[int, int]]:
    parts = _split_query_tokens(query)
    if not parts:
        return
    needles = [_fold(_lemma_of(p, language), case_sensitive=case_sensitive) for p in parts]
    n = len(needles)
    lemmas = [_fold(_lemma_of(tok.text, language), case_sensitive=case_sensitive) for tok in tokens]
    for start in range(0, len(lemmas) - n + 1):
        if lemmas[start : start + n] == needles:
            yield start, start + n


def _build_hit(
    text: str,
    tokens: list[TokenSpan],
    token_start: int,
    token_end: int,
    *,
    window: int,
    meta: dict[str, Any],
    query: str,
    query_mode: str,
    case_sensitive: bool,
    match_char_start: int | None = None,
    match_char_end: int | None = None,
) -> dict[str, Any]:
    left_tokens = [t.text for t in tokens[max(0, token_start - window) : token_start]]
    match_tokens = [t.text for t in tokens[token_start:token_end]]
    right_tokens = [t.text for t in tokens[token_end : token_end + window]]

    if match_char_start is None:
        match_char_start = tokens[token_start].start if tokens and token_start < len(tokens) else 0
    if match_char_end is None:
        match_char_end = (
            tokens[token_end - 1].end if tokens and token_end > token_start else match_char_start
        )

    unit_char_start = meta.get("unit_char_start")
    absolute_start = (
        int(unit_char_start) + match_char_start if isinstance(unit_char_start, int) else None
    )
    absolute_end = (
        int(unit_char_start) + match_char_end if isinstance(unit_char_start, int) else None
    )

    hit: dict[str, Any] = {
        "left_context": " ".join(left_tokens),
        "keyword": " ".join(match_tokens) if match_tokens else text[match_char_start:match_char_end],
        "right_context": " ".join(right_tokens),
        "left_tokens": left_tokens,
        "match_tokens": match_tokens,
        "right_tokens": right_tokens,
        "token_start": token_start,
        "token_end": token_end,
        "match_char_start": match_char_start,
        "match_char_end": match_char_end,
        "absolute_char_start": absolute_start,
        "absolute_char_end": absolute_end,
        "context_width": window,
        "context_unit": "token",
        "query": query,
        "query_mode": query_mode,
        "case_sensitive": case_sensitive,
    }
    # Provenance / source metadata from the caller (unit + document fields).
    for key, value in meta.items():
        if key == "text":
            continue
        hit.setdefault(key, value)
    return hit


def concordance(
    texts: list[str],
    metadata: list[dict[str, Any]],
    query: str,
    *,
    window: int = 5,
    case_sensitive: bool = False,
    query_mode: str | None = "auto",
    language: str | None = None,
    token_attribute: str | None = None,
    max_matches: int | None = None,
) -> list[dict[str, Any]]:
    """Search units for ``query`` and return KWIC / concordance hits.

    Context is always **token-based** (``window`` tokens on each side). Source and
    document/page/unit provenance are copied from each metadata dict.
    """
    if len(texts) != len(metadata):
        raise ValueError("texts and metadata must have the same length")
    if window < 0:
        raise ValueError("window must be >= 0")
    if not str(query).strip():
        raise ValueError("query must be non-empty")

    mode = normalize_query_mode(query_mode, query)
    attr = (token_attribute or "surface").strip().lower()
    if attr not in SUPPORTED_TOKEN_ATTRIBUTES:
        raise ValueError(
            f"Unsupported token_attribute {token_attribute!r}; "
            f"supported: {', '.join(sorted(SUPPORTED_TOKEN_ATTRIBUTES))}"
        )
    if attr == "lemma":
        mode = "lemma"

    results: list[dict[str, Any]] = []

    for text, meta in zip(texts, metadata, strict=True):
        tokens = tokenize_with_spans(text)
        spans: list[tuple[int, int, int | None, int | None]] = []

        if mode == "word":
            for a, b in _iter_word_matches(tokens, query, case_sensitive=case_sensitive):
                spans.append((a, b, None, None))
        elif mode == "phrase":
            for a, b in _iter_phrase_matches(tokens, query, case_sensitive=case_sensitive):
                spans.append((a, b, None, None))
        elif mode == "exact_phrase":
            for c0, c1 in _iter_exact_phrase_char_matches(
                text, query, case_sensitive=case_sensitive
            ):
                mapped = _char_span_to_token_span(tokens, c0, c1)
                if mapped is None:
                    continue
                spans.append((*mapped, c0, c1))
        elif mode == "regex":
            for c0, c1 in _iter_regex_matches(text, query, case_sensitive=case_sensitive):
                mapped = _char_span_to_token_span(tokens, c0, c1)
                if mapped is None:
                    continue
                spans.append((*mapped, c0, c1))
        elif mode == "wildcard":
            for a, b in _iter_wildcard_matches(tokens, query, case_sensitive=case_sensitive):
                spans.append((a, b, None, None))
        elif mode == "lemma":
            for a, b in _iter_lemma_matches(
                tokens, query, case_sensitive=case_sensitive, language=language
            ):
                spans.append((a, b, None, None))
        else:  # pragma: no cover
            raise ValueError(f"Unsupported query mode {mode!r}")

        for token_start, token_end, c0, c1 in spans:
            results.append(
                _build_hit(
                    text,
                    tokens,
                    token_start,
                    token_end,
                    window=window,
                    meta=meta,
                    query=query,
                    query_mode=mode,
                    case_sensitive=case_sensitive,
                    match_char_start=c0,
                    match_char_end=c1,
                )
            )
            if max_matches is not None and len(results) >= max_matches:
                return results

    return results


def kwic(
    texts: list[str],
    metadata: list[dict[str, Any]],
    keyword: str,
    window: int = 5,
    *,
    case_sensitive: bool = False,
    query_mode: str | None = "auto",
    language: str | None = None,
    token_attribute: str | None = None,
    max_matches: int | None = None,
) -> list[dict[str, Any]]:
    """Backward-compatible alias for :func:`concordance`."""
    return concordance(
        texts,
        metadata,
        keyword,
        window=window,
        case_sensitive=case_sensitive,
        query_mode=query_mode,
        language=language,
        token_attribute=token_attribute,
        max_matches=max_matches,
    )


def kwic_search(
    payload: list[dict[str, Any]],
    keyword: str,
    *,
    window_size: int = 5,
    case_sensitive: bool = False,
    query_mode: str | None = "auto",
    language: str | None = None,
    token_attribute: str | None = None,
    max_matches: int | None = None,
) -> list[dict[str, Any]]:
    """KWIC over payload rows that each contain ``text`` plus provenance fields."""
    texts = [row["text"] for row in payload]
    metadata = [{key: value for key, value in row.items() if key != "text"} for row in payload]
    return concordance(
        texts,
        metadata,
        keyword,
        window=window_size,
        case_sensitive=case_sensitive,
        query_mode=query_mode,
        language=language,
        token_attribute=token_attribute,
        max_matches=max_matches,
    )


def describe_kwic_capabilities() -> dict[str, Any]:
    return {
        "query_modes": sorted(QUERY_MODES - {"auto"}),
        "auto_resolution": "wildcard if *|?; phrase if whitespace; else word",
        "context_unit": "token",
        "token_attributes": sorted(SUPPORTED_TOKEN_ATTRIBUTES),
        "case_sensitive": True,
        "case_insensitive": True,
        "provenance_fields": [
            "text_unit_id",
            "corpus_document_id",
            "page_number",
            "section_heading",
            "unit_char_start",
            "unit_char_end",
            "match_char_start",
            "match_char_end",
            "absolute_char_start",
            "absolute_char_end",
            "document_title",
            "organization",
            "publication_year",
        ],
    }
