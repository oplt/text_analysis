"""Deterministic page-level quality signals for native PDF text extraction."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PdfTextQuality:
    score: float
    printable_ratio: float
    replacement_ratio: float
    character_count: int
    is_low_quality: bool


@dataclass(frozen=True, slots=True)
class OcrDecision:
    should_ocr: bool
    reason: str
    quality: PdfTextQuality


def assess_pdf_text_quality(text: str, *, threshold: float = 0.65) -> PdfTextQuality:
    text = text or ""
    character_count = len(text)
    if not character_count:
        return PdfTextQuality(0.0, 0.0, 0.0, 0, True)
    printable_ratio = sum(char.isprintable() or char.isspace() for char in text) / character_count
    replacement_ratio = text.count("\ufffd") / character_count
    # Very short native extracts are a practical scanned-page signal. The
    # score deliberately remains inspectable rather than pretending to detect
    # every layout defect.
    density_score = min(1.0, character_count / 80)
    score = max(0.0, min(1.0, printable_ratio * density_score * (1 - replacement_ratio)))
    return PdfTextQuality(
        score=score,
        printable_ratio=printable_ratio,
        replacement_ratio=replacement_ratio,
        character_count=character_count,
        is_low_quality=score < threshold,
    )


def decide_ocr(
    text: str,
    *,
    ocr_enabled: bool = False,
    quality_threshold: float = 0.65,
    min_text_chars: int = 40,
) -> OcrDecision:
    """Conservative OCR gate: never OCR by default; only flag poor native text."""
    quality = assess_pdf_text_quality(text, threshold=quality_threshold)
    if not ocr_enabled:
        return OcrDecision(False, "ocr_disabled", quality)
    if quality.character_count < max(1, min_text_chars) or quality.is_low_quality:
        return OcrDecision(True, "low_quality_or_sparse_native_text", quality)
    return OcrDecision(False, "native_text_sufficient", quality)


def suppress_repeated_headers_footers(
    page_texts: list[str],
    *,
    min_pages: int = 3,
    min_repetition_ratio: float = 0.6,
) -> list[str]:
    """Drop short lines that repeat across most pages (simple header/footer heuristic)."""
    if len(page_texts) < min_pages:
        return list(page_texts)

    line_page_counts: Counter[str] = Counter()
    for text in page_texts:
        unique_lines = {
            line.strip()
            for line in (text or "").splitlines()
            if line.strip() and len(line.strip()) <= 120
        }
        for line in unique_lines:
            line_page_counts[line] += 1

    threshold = max(2, int(len(page_texts) * min_repetition_ratio))
    repeated = {line for line, count in line_page_counts.items() if count >= threshold}
    if not repeated:
        return list(page_texts)

    cleaned: list[str] = []
    for text in page_texts:
        kept = [line for line in (text or "").splitlines() if line.strip() not in repeated]
        cleaned.append("\n".join(kept).strip())
    return cleaned
