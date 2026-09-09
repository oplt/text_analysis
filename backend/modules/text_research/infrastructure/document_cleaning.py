"""Configurable document-level cleaning (raw extract → cleaned canonical).

All transforms are optional, recorded, and reproducible. Input text is never
mutated in place; discarded spans are summarized in step metadata so nothing
is silently thrown away without an audit trail.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any

from backend.modules.text_research.infrastructure.canonical_text import sha256_text

CLEANING_ENGINE = "text_research.document_cleaning"
CLEANING_ENGINE_VERSION = "1"

DEFAULT_CLEANING_CONFIG: dict[str, Any] = {
    "unicode_normalization": None,  # None | NFC | NFKC | NFD | NFKD
    "normalize_whitespace": False,
    "dehyphenate_line_breaks": False,
    "remove_headers": False,
    "remove_footers": False,
    "remove_page_numbers": False,
    "exclude_bibliography": False,
    "exclude_tables": False,
    "custom_regex": [],
}

_VALID_UNICODE_FORMS = frozenset({"NFC", "NFKC", "NFD", "NFKD"})
_HYPHEN_BREAK_RE = re.compile(r"([A-Za-z])-\s*\n\s*([A-Za-z])")
_PAGE_NUMBER_LINE_RE = re.compile(
    r"^\s*(?:page\s+)?(?:\d{1,4}|[-–—]\s*\d{1,4}\s*[-–—])\s*$",
    re.IGNORECASE,
)
_BIBLIOGRAPHY_HEADING_RE = re.compile(
    r"^\s*(references|bibliography|works\s+cited|literature\s+cited|"
    r"reference\s+list|endnotes)\s*:?\s*$",
    re.IGNORECASE,
)
_TABLE_PIPE_RE = re.compile(r"^\s*\|.+\|\s*$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:-]+\|[\s|:-]*$")
_DISCARD_PREVIEW_CHARS = 240
_HEADER_MIN_LEN = 8
_HEADER_MIN_REPEATS = 3


@dataclass
class CleaningConfig:
    unicode_normalization: str | None = None
    normalize_whitespace: bool = False
    dehyphenate_line_breaks: bool = False
    remove_headers: bool = False
    remove_footers: bool = False
    remove_page_numbers: bool = False
    exclude_bibliography: bool = False
    exclude_tables: bool = False
    custom_regex: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> CleaningConfig:
        merged = dict(DEFAULT_CLEANING_CONFIG)
        if data:
            merged.update(data)
        form = merged.get("unicode_normalization")
        if form is not None:
            form = str(form).strip().upper()
            if form not in _VALID_UNICODE_FORMS:
                raise ValueError(
                    f"unicode_normalization must be one of "
                    f"{sorted(_VALID_UNICODE_FORMS)} or null, got {form!r}"
                )
            merged["unicode_normalization"] = form
        custom = merged.get("custom_regex") or []
        if not isinstance(custom, list):
            raise ValueError("custom_regex must be a list of {pattern, replacement} objects")
        normalized_custom: list[dict[str, Any]] = []
        for i, item in enumerate(custom):
            if not isinstance(item, dict) or "pattern" not in item:
                raise ValueError(f"custom_regex[{i}] must be an object with pattern")
            try:
                re.compile(str(item["pattern"]))
            except re.error as exc:
                raise ValueError(f"custom_regex[{i}] invalid pattern: {exc}") from exc
            normalized_custom.append(
                {
                    "pattern": str(item["pattern"]),
                    "replacement": str(item.get("replacement", "")),
                    "count": int(item["count"]) if item.get("count") is not None else 0,
                }
            )
        merged["custom_regex"] = normalized_custom
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: merged[k] for k in field_names if k in merged})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CleaningStepRecord:
    name: str
    enabled: bool
    chars_before: int
    chars_after: int
    discarded_chars: int = 0
    discarded_preview: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CleaningResult:
    cleaned_text: str
    raw_text: str
    raw_checksum: str
    cleaned_checksum: str
    steps: list[CleaningStepRecord]
    config: dict[str, Any]
    engine: str = CLEANING_ENGINE
    engine_version: str = CLEANING_ENGINE_VERSION

    def to_transformation_metadata(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "engine_version": self.engine_version,
            "config": self.config,
            "raw_checksum": self.raw_checksum,
            "cleaned_checksum": self.cleaned_checksum,
            "chars_raw": len(self.raw_text),
            "chars_cleaned": len(self.cleaned_text),
            "steps": [s.to_dict() for s in self.steps],
        }


def _preview_discard(before: str, after: str) -> tuple[int, str | None]:
    discarded = len(before) - len(after)
    if discarded <= 0:
        return 0, None
    # Prefer a short slice of material that disappeared (best-effort).
    if after in before:
        idx = before.find(after)
        removed = (before[:idx] + before[idx + len(after) :])[:_DISCARD_PREVIEW_CHARS]
    else:
        removed = before[:_DISCARD_PREVIEW_CHARS]
    return discarded, removed or None


def _record_step(
    name: str,
    *,
    enabled: bool,
    before: str,
    after: str,
    details: dict[str, Any] | None = None,
) -> CleaningStepRecord:
    discarded, preview = _preview_discard(before, after) if enabled else (0, None)
    return CleaningStepRecord(
        name=name,
        enabled=enabled,
        chars_before=len(before),
        chars_after=len(after) if enabled else len(before),
        discarded_chars=discarded if enabled else 0,
        discarded_preview=preview if enabled else None,
        details=details or {},
    )


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def _dehyphenate(text: str) -> str:
    return _HYPHEN_BREAK_RE.sub(r"\1\2", text)


def _repeated_edge_lines(paragraphs: list[str], *, edge: str) -> set[str]:
    lines: list[str] = []
    for para in paragraphs:
        parts = [ln.strip() for ln in para.splitlines() if ln.strip()]
        if not parts:
            continue
        candidate = parts[0] if edge == "head" else parts[-1]
        if len(candidate) >= _HEADER_MIN_LEN:
            lines.append(candidate)
    counts: dict[str, int] = {}
    for line in lines:
        counts[line] = counts.get(line, 0) + 1
    return {line for line, count in counts.items() if count >= _HEADER_MIN_REPEATS}


def _remove_edge_lines(text: str, *, remove_heads: bool, remove_tails: bool) -> tuple[str, dict]:
    paragraphs = [p for p in re.split(r"\n\s*\n+", text) if p.strip()]
    if len(paragraphs) < 4:
        return text, {"skipped": True, "reason": "too_few_paragraphs"}

    heads = _repeated_edge_lines(paragraphs, edge="head") if remove_heads else set()
    tails = _repeated_edge_lines(paragraphs, edge="tail") if remove_tails else set()
    if not heads and not tails:
        return text, {"removed_headers": [], "removed_footers": []}

    cleaned_paras: list[str] = []
    removed_h: list[str] = []
    removed_f: list[str] = []
    for para in paragraphs:
        lines = para.splitlines()
        if lines and heads and lines[0].strip() in heads:
            removed_h.append(lines[0].strip())
            lines = lines[1:]
        if lines and tails and lines[-1].strip() in tails:
            removed_f.append(lines[-1].strip())
            lines = lines[:-1]
        rebuilt = "\n".join(lines).strip()
        if rebuilt:
            cleaned_paras.append(rebuilt)
    return "\n\n".join(cleaned_paras), {
        "removed_headers": sorted(set(removed_h))[:20],
        "removed_footers": sorted(set(removed_f))[:20],
    }


def _remove_page_numbers(text: str) -> tuple[str, dict]:
    kept: list[str] = []
    removed = 0
    for line in text.splitlines():
        if _PAGE_NUMBER_LINE_RE.match(line):
            removed += 1
            continue
        kept.append(line)
    return "\n".join(kept), {"removed_line_count": removed}


def _exclude_bibliography(text: str) -> tuple[str, dict]:
    lines = text.splitlines()
    cut_at: int | None = None
    heading: str | None = None
    for i, line in enumerate(lines):
        # Only cut if enough body remains before the heading.
        if _BIBLIOGRAPHY_HEADING_RE.match(line) and i >= 3:
            cut_at = i
            heading = line.strip()
            break
    if cut_at is None:
        return text, {"cut": False}
    kept = "\n".join(lines[:cut_at]).rstrip()
    discarded = "\n".join(lines[cut_at:])
    return kept, {
        "cut": True,
        "heading": heading,
        "discarded_chars": len(discarded),
        "discarded_preview": discarded[:_DISCARD_PREVIEW_CHARS],
    }


def _exclude_tables(text: str) -> tuple[str, dict]:
    kept: list[str] = []
    removed_lines = 0
    for line in text.splitlines():
        if _TABLE_PIPE_RE.match(line) or _TABLE_SEP_RE.match(line):
            removed_lines += 1
            continue
        # Dense tab-separated row heuristic
        if "\t" in line and line.count("\t") >= 3 and len(line.split("\t")) >= 4:
            removed_lines += 1
            continue
        kept.append(line)
    return "\n".join(kept), {"removed_line_count": removed_lines}


def _apply_custom_regex(text: str, rules: list[dict[str, Any]]) -> tuple[str, dict]:
    current = text
    applied: list[dict[str, Any]] = []
    for rule in rules:
        pattern = rule["pattern"]
        replacement = rule.get("replacement", "")
        count = int(rule.get("count") or 0)
        before = current
        if count > 0:
            current, n = re.subn(pattern, replacement, current, count=count)
        else:
            current, n = re.subn(pattern, replacement, current)
        applied.append(
            {
                "pattern": pattern,
                "replacement": replacement,
                "substitutions": n,
                "chars_delta": len(current) - len(before),
            }
        )
    return current, {"rules": applied}


def apply_cleaning(
    raw_text: str, config: CleaningConfig | dict[str, Any] | None = None
) -> CleaningResult:
    """Apply optional cleaning steps to a copy of ``raw_text``.

    Original ``raw_text`` is never modified. Discarded content is summarized in
    step records (char counts + short preview).
    """
    if isinstance(config, dict) or config is None:
        resolved = CleaningConfig.from_dict(config)
    else:
        resolved = config

    raw = raw_text  # do not mutate caller
    current = raw
    steps: list[CleaningStepRecord] = []

    # Unicode normalization
    form = resolved.unicode_normalization
    before = current
    if form:
        current = unicodedata.normalize(form, current)
    steps.append(
        _record_step(
            "unicode_normalization",
            enabled=bool(form),
            before=before,
            after=current,
            details={"form": form},
        )
    )

    # Dehyphenation (before whitespace collapse so linebreak context remains)
    before = current
    if resolved.dehyphenate_line_breaks:
        current = _dehyphenate(current)
    steps.append(
        _record_step(
            "dehyphenate_line_breaks",
            enabled=resolved.dehyphenate_line_breaks,
            before=before,
            after=current,
        )
    )

    # Whitespace
    before = current
    if resolved.normalize_whitespace:
        current = _normalize_whitespace(current)
    steps.append(
        _record_step(
            "normalize_whitespace",
            enabled=resolved.normalize_whitespace,
            before=before,
            after=current,
        )
    )

    # Headers / footers
    before = current
    details: dict[str, Any] = {}
    if resolved.remove_headers or resolved.remove_footers:
        current, details = _remove_edge_lines(
            current,
            remove_heads=resolved.remove_headers,
            remove_tails=resolved.remove_footers,
        )
    steps.append(
        _record_step(
            "remove_headers_footers",
            enabled=resolved.remove_headers or resolved.remove_footers,
            before=before,
            after=current,
            details={
                **details,
                "remove_headers": resolved.remove_headers,
                "remove_footers": resolved.remove_footers,
            },
        )
    )

    # Page numbers
    before = current
    page_details: dict[str, Any] = {}
    if resolved.remove_page_numbers:
        current, page_details = _remove_page_numbers(current)
    steps.append(
        _record_step(
            "remove_page_numbers",
            enabled=resolved.remove_page_numbers,
            before=before,
            after=current,
            details=page_details,
        )
    )

    # Bibliography
    before = current
    bib_details: dict[str, Any] = {}
    if resolved.exclude_bibliography:
        current, bib_details = _exclude_bibliography(current)
    steps.append(
        _record_step(
            "exclude_bibliography",
            enabled=resolved.exclude_bibliography,
            before=before,
            after=current,
            details=bib_details,
        )
    )

    # Tables
    before = current
    table_details: dict[str, Any] = {}
    if resolved.exclude_tables:
        current, table_details = _exclude_tables(current)
    steps.append(
        _record_step(
            "exclude_tables",
            enabled=resolved.exclude_tables,
            before=before,
            after=current,
            details=table_details,
        )
    )

    # Custom regex
    before = current
    regex_details: dict[str, Any] = {}
    if resolved.custom_regex:
        current, regex_details = _apply_custom_regex(current, resolved.custom_regex)
    steps.append(
        _record_step(
            "custom_regex",
            enabled=bool(resolved.custom_regex),
            before=before,
            after=current,
            details=regex_details,
        )
    )

    return CleaningResult(
        cleaned_text=current,
        raw_text=raw,
        raw_checksum=sha256_text(raw),
        cleaned_checksum=sha256_text(current),
        steps=steps,
        config=resolved.to_dict(),
    )


def preview_cleaning(
    texts: list[str],
    config: CleaningConfig | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Preview cleaning on sample strings without persistence."""
    resolved = CleaningConfig.from_dict(
        config if isinstance(config, dict) or config is None else config.to_dict()
    )
    rows: list[dict[str, Any]] = []
    for text in texts:
        result = apply_cleaning(text, resolved)
        rows.append(
            {
                "raw_preview": text[:400],
                "cleaned_preview": result.cleaned_text[:400],
                "chars_raw": len(text),
                "chars_cleaned": len(result.cleaned_text),
                "raw_checksum": result.raw_checksum,
                "cleaned_checksum": result.cleaned_checksum,
                "steps": [s.to_dict() for s in result.steps if s.enabled],
            }
        )
    return {
        "rows": rows,
        "config": resolved.to_dict(),
        "engine": CLEANING_ENGINE,
        "engine_version": CLEANING_ENGINE_VERSION,
    }
