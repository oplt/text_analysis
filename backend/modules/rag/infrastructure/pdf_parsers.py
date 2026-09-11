"""PDF parsing adapters: basic pypdf + optional enhanced layout-aware parser."""

from __future__ import annotations

import logging
from typing import Protocol

from backend.modules.rag.domain.models import ParsedDocument

logger = logging.getLogger(__name__)


class PdfParserAdapter(Protocol):
    def parse(self, content: bytes) -> list[ParsedDocument]: ...


class BasicPypdfParser:
    """Baseline PDF parser using pypdf page text extraction."""

    name = "pypdf-v1"

    def parse(self, content: bytes) -> list[ParsedDocument]:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError(
                "pypdf is required for PDF parsing. Install with: uv add pypdf"
            ) from exc

        import io

        reader = PdfReader(io.BytesIO(content))
        docs: list[ParsedDocument] = []
        for index, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                docs.append(
                    ParsedDocument(
                        content=text,
                        metadata={
                            "format": "pdf",
                            "page_number": index,
                            "parser": self.name,
                        },
                        page_number=index,
                    )
                )
        return docs


class EnhancedPdfParser:
    """Optional layout-aware parser.

    Attempts ``pymupdf`` (fitz) when installed; callers must fall back to basic
    parser when this raises or returns empty.
    """

    name = "pymupdf-v1"

    def parse(self, content: bytes) -> list[ParsedDocument]:
        try:
            import fitz  # pymupdf
        except ImportError as exc:
            raise RuntimeError("enhanced PDF parser unavailable") from exc

        docs: list[ParsedDocument] = []
        with fitz.open(stream=content, filetype="pdf") as document:
            for index, page in enumerate(document, start=1):
                text = (page.get_text("text") or "").strip()
                # Soft-dehyphenate line-break hyphens common in academic PDFs
                text = text.replace("-\n", "")
                if text:
                    docs.append(
                        ParsedDocument(
                            content=text,
                            metadata={
                                "format": "pdf",
                                "page_number": index,
                                "parser": self.name,
                            },
                            page_number=index,
                        )
                    )
        if not docs:
            raise RuntimeError("enhanced PDF parser returned no text")
        return docs


def parse_pdf_with_fallback(content: bytes) -> list[ParsedDocument]:
    """Prefer enhanced parser when available; always fall back to pypdf."""
    try:
        return EnhancedPdfParser().parse(content)
    except Exception:
        logger.info("Enhanced PDF parser unavailable or failed; using pypdf baseline")
        return BasicPypdfParser().parse(content)
