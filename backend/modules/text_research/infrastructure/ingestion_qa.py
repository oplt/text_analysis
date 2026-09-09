"""Document ingestion quality diagnostics (pure functions; never mutate text).

Findings are informational. Callers persist reports separately and must not
rewrite canonical/raw extracts based on these checks alone.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

from backend.modules.text_research.infrastructure.canonical_text import sha256_text
from backend.modules.text_research.infrastructure.duplicate_detection import (
    duplicate_report,
    group_by_key,
)

# Tunable defaults — generic, not domain-specific.
SHORT_CHAR_THRESHOLD = 40
LARGE_CHAR_THRESHOLD = 2_000_000
REPLACEMENT_RATIO_WARN = 0.01
REPLACEMENT_ABS_WARN = 5
WHITESPACE_RATIO_WARN = 0.45
CONTROL_CHAR_ABS_WARN = 3
HYPHEN_BREAK_ABS_WARN = 8
NEAR_DUPLICATE_JACCARD = 0.92
TOKEN_OUTLIER_Z = 3.0
HEADER_REPEAT_MIN_DOCS = 3
SCANNED_PDF_LETTER_RATIO = 0.15
SCANNED_PDF_MAX_CHARS = 80

_TOKEN_RE = re.compile(r"\S+")
_LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)
_HYPHEN_BREAK_RE = re.compile(r"[A-Za-z]-\s*\n\s*[A-Za-z]")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_LONG_WS_RE = re.compile(r"[ \t]{4,}|\n{4,}")


@dataclass(slots=True)
class QaFinding:
    code: str
    severity: str  # info | warning | error
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def tokenize_rough(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


# Character n-gram / multiset-Jaccard helpers now live in
# :mod:`duplicate_detection` (see ``detect_near_duplicates`` below) so
# ingestion QA and the explicit duplicate-detection analysis always agree on
# the near-duplicate metric.


def analyze_document_text(
    text: str,
    *,
    language: str | None = None,
    filename: str | None = None,
    content_type: str | None = None,
    has_extraction_failure: bool = False,
) -> tuple[list[QaFinding], dict[str, Any]]:
    """Run per-document diagnostics. Does not modify ``text``."""
    findings: list[QaFinding] = []
    stripped = text.strip()
    char_count = len(text)
    token_count = len(tokenize_rough(text))
    letter_count = len(_LETTER_RE.findall(text))
    replacement_count = text.count("\ufffd")
    whitespace_count = sum(1 for ch in text if ch.isspace())
    control_count = len(_CONTROL_RE.findall(text))
    hyphen_breaks = len(_HYPHEN_BREAK_RE.findall(text))
    letter_ratio = (letter_count / char_count) if char_count else 0.0
    whitespace_ratio = (whitespace_count / char_count) if char_count else 0.0
    replacement_ratio = (replacement_count / char_count) if char_count else 0.0

    metrics: dict[str, Any] = {
        "char_count": char_count,
        "token_count": token_count,
        "letter_count": letter_count,
        "letter_ratio": round(letter_ratio, 4),
        "whitespace_ratio": round(whitespace_ratio, 4),
        "replacement_char_count": replacement_count,
        "replacement_ratio": round(replacement_ratio, 4),
        "control_char_count": control_count,
        "hyphen_linebreak_count": hyphen_breaks,
        "text_checksum": sha256_text(text),
        "text_mutated": False,
    }

    if has_extraction_failure or not stripped:
        findings.append(
            QaFinding(
                code="empty_document" if not stripped else "extraction_failure",
                severity="error",
                message=(
                    "Document text is empty."
                    if not stripped
                    else "Extraction reported failure and produced little or no usable text."
                ),
                details={"char_count": char_count},
            )
        )
        if not stripped:
            return findings, metrics

    if has_extraction_failure and stripped:
        findings.append(
            QaFinding(
                code="extraction_failure",
                severity="error",
                message="Extraction was marked as failed for this document.",
            )
        )

    if 0 < char_count < SHORT_CHAR_THRESHOLD:
        findings.append(
            QaFinding(
                code="suspiciously_short",
                severity="warning",
                message=f"Document is unusually short (< {SHORT_CHAR_THRESHOLD} characters).",
                details={"char_count": char_count, "threshold": SHORT_CHAR_THRESHOLD},
            )
        )

    if char_count > LARGE_CHAR_THRESHOLD:
        findings.append(
            QaFinding(
                code="suspiciously_large",
                severity="warning",
                message=f"Document is unusually large (> {LARGE_CHAR_THRESHOLD} characters).",
                details={"char_count": char_count, "threshold": LARGE_CHAR_THRESHOLD},
            )
        )

    if replacement_count >= REPLACEMENT_ABS_WARN or replacement_ratio >= REPLACEMENT_RATIO_WARN:
        findings.append(
            QaFinding(
                code="excessive_replacement_chars",
                severity="warning",
                message="Excessive Unicode replacement characters (U+FFFD) — possible encoding damage.",
                details={
                    "replacement_char_count": replacement_count,
                    "replacement_ratio": round(replacement_ratio, 4),
                },
            )
        )

    looks_pdf = bool(
        (filename and filename.lower().endswith(".pdf"))
        or (content_type and "pdf" in content_type.lower())
    )
    if looks_pdf and (
        char_count <= SCANNED_PDF_MAX_CHARS or letter_ratio < SCANNED_PDF_LETTER_RATIO
    ):
        findings.append(
            QaFinding(
                code="likely_image_only_or_scanned_pdf",
                severity="warning",
                message="PDF yields little extractable text — may be image-only or poorly OCR'd.",
                details={"char_count": char_count, "letter_ratio": round(letter_ratio, 4)},
            )
        )

    if whitespace_ratio >= WHITESPACE_RATIO_WARN or _LONG_WS_RE.search(text):
        findings.append(
            QaFinding(
                code="unusual_whitespace",
                severity="info",
                message="Unusual whitespace density or long whitespace runs detected.",
                details={"whitespace_ratio": round(whitespace_ratio, 4)},
            )
        )

    if hyphen_breaks >= HYPHEN_BREAK_ABS_WARN:
        findings.append(
            QaFinding(
                code="line_break_hyphenation",
                severity="info",
                message="Frequent end-of-line hyphenation patterns detected.",
                details={"hyphen_linebreak_count": hyphen_breaks},
            )
        )

    if control_count >= CONTROL_CHAR_ABS_WARN or "\x00" in text:
        findings.append(
            QaFinding(
                code="malformed_extraction",
                severity="warning",
                message="Control characters or null bytes suggest malformed extraction.",
                details={"control_char_count": control_count, "has_null": "\x00" in text},
            )
        )

    if language:
        lang = language.strip().lower()
        ascii_letters = sum(1 for ch in text if ("a" <= ch.lower() <= "z"))
        ascii_ratio = ascii_letters / char_count if char_count else 0.0
        if lang.startswith("en") and ascii_ratio < 0.5 and letter_ratio > 0.2:
            findings.append(
                QaFinding(
                    code="language_mismatch",
                    severity="warning",
                    message="Language metadata suggests English, but character distribution looks non-English.",
                    details={"language": language, "ascii_letter_ratio": round(ascii_ratio, 4)},
                )
            )
        if lang and not lang.startswith("en") and ascii_ratio > 0.85 and letter_ratio > 0.4:
            findings.append(
                QaFinding(
                    code="language_mismatch",
                    severity="info",
                    message="Language metadata is non-English, but text is overwhelmingly Latin/ASCII.",
                    details={"language": language, "ascii_letter_ratio": round(ascii_ratio, 4)},
                )
            )

    # Repeated header/footer heuristics on paragraph first/last lines.
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    if len(paragraphs) >= 4:
        heads = [p.splitlines()[0].strip() for p in paragraphs if p.splitlines()]
        tails = [p.splitlines()[-1].strip() for p in paragraphs if p.splitlines()]
        head_counts = Counter(h for h in heads if len(h) >= 8)
        tail_counts = Counter(t for t in tails if len(t) >= 8)
        for line, count in head_counts.most_common(3):
            if count >= HEADER_REPEAT_MIN_DOCS:
                findings.append(
                    QaFinding(
                        code="repeated_headers",
                        severity="info",
                        message="Possible repeated header line across paragraphs.",
                        details={"line": line[:200], "count": count},
                    )
                )
                break
        for line, count in tail_counts.most_common(3):
            if count >= HEADER_REPEAT_MIN_DOCS:
                findings.append(
                    QaFinding(
                        code="repeated_footers",
                        severity="info",
                        message="Possible repeated footer line across paragraphs.",
                        details={"line": line[:200], "count": count},
                    )
                )
                break

    return findings, metrics


def detect_exact_duplicates(
    documents: list[dict[str, Any]],
) -> list[QaFinding]:
    """``documents`` items: {document_id, checksum}.

    Delegates grouping to :func:`duplicate_detection.group_by_key` so ingestion
    QA and the explicit duplicate-detection analysis share one implementation.
    """
    rows = [
        (doc["document_id"], doc["checksum"])
        for doc in documents
        if doc.get("checksum") and doc.get("document_id")
    ]
    findings: list[QaFinding] = []
    for group in group_by_key(rows):
        findings.append(
            QaFinding(
                code="exact_duplicate",
                severity="warning",
                message="Exact duplicate canonical text across documents.",
                details={"checksum": group["key"], "document_ids": group["ids"]},
            )
        )
    return findings


def detect_near_duplicates(
    documents: list[dict[str, Any]],
    *,
    threshold: float = NEAR_DUPLICATE_JACCARD,
) -> list[QaFinding]:
    """``documents`` items: {document_id, text}. Pairwise char-ngram Jaccard.

    Delegates to :func:`duplicate_detection.duplicate_report` so ingestion QA
    and the explicit "lexical" duplicate-detection analysis always agree.
    """
    items = [
        {"id": doc["document_id"], "text": doc.get("text") or ""}
        for doc in documents
        if doc.get("document_id") and (doc.get("text") or "").strip()
    ]
    report = duplicate_report(items, methods=["lexical"], lexical_threshold=threshold)
    findings: list[QaFinding] = []
    for pair in report["lexical_near_duplicates"]:
        findings.append(
            QaFinding(
                code="likely_near_duplicate",
                severity="warning",
                message="Likely near-duplicate documents (high character n-gram overlap).",
                details={
                    "document_ids": [pair["source_id"], pair["target_id"]],
                    "jaccard": pair["score"],
                    "threshold": threshold,
                },
            )
        )
    return findings


def detect_token_count_outliers(
    documents: list[dict[str, Any]],
    *,
    z_threshold: float = TOKEN_OUTLIER_Z,
) -> list[QaFinding]:
    """``documents`` items: {document_id, token_count}."""
    counts = [int(doc["token_count"]) for doc in documents if "token_count" in doc]
    if len(counts) < 5:
        return []
    mean = sum(counts) / len(counts)
    variance = sum((c - mean) ** 2 for c in counts) / len(counts)
    std = math.sqrt(variance)
    if std == 0:
        return []
    findings: list[QaFinding] = []
    for doc in documents:
        token_count = int(doc.get("token_count") or 0)
        z = abs(token_count - mean) / std
        if z >= z_threshold:
            findings.append(
                QaFinding(
                    code="extreme_token_count_outlier",
                    severity="info",
                    message="Document token count is an extreme corpus outlier.",
                    details={
                        "document_id": doc.get("document_id"),
                        "token_count": token_count,
                        "z_score": round(z, 3),
                        "mean": round(mean, 2),
                        "std": round(std, 2),
                    },
                )
            )
    return findings
