"""PDF parsing adapters: pypdf baseline + PyMuPDF block/reading-order extraction."""

from __future__ import annotations

import logging
from typing import Protocol
from uuid import uuid4

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
                            "parser_quality": "basic",
                        },
                        page_number=index,
                    )
                )
        return docs


class EnhancedPdfParser:
    """Layout-aware parser with block/bbox reading-order metadata when pymupdf is available."""

    name = "pymupdf-blocks-v1"

    def parse(self, content: bytes) -> list[ParsedDocument]:
        try:
            import fitz  # pymupdf
        except ImportError as exc:
            raise RuntimeError("enhanced PDF parser unavailable") from exc

        docs: list[ParsedDocument] = []
        with fitz.open(stream=content, filetype="pdf") as document:
            for index, page in enumerate(document, start=1):
                blocks = page.get_text("dict").get("blocks") or []
                text_blocks: list[dict] = []
                lines_out: list[str] = []
                for block in blocks:
                    if block.get("type") != 0:
                        continue
                    block_id = str(uuid4())
                    bbox = block.get("bbox")
                    parts: list[str] = []
                    for line in block.get("lines") or []:
                        span_text = "".join(
                            span.get("text") or "" for span in (line.get("spans") or [])
                        ).strip()
                        if span_text:
                            parts.append(span_text)
                    block_text = "\n".join(parts).strip()
                    if not block_text:
                        continue
                    # Soft-dehyphenate line-break hyphens common in academic PDFs
                    block_text = block_text.replace("-\n", "")
                    text_blocks.append(
                        {
                            "block_id": block_id,
                            "bbox": list(bbox) if bbox else None,
                            "text": block_text,
                        }
                    )
                    lines_out.append(block_text)

                text = "\n\n".join(lines_out).strip()
                if not text:
                    # Fall back to plain text extraction for the page.
                    text = (page.get_text("text") or "").strip().replace("-\n", "")
                if text:
                    docs.append(
                        ParsedDocument(
                            content=text,
                            metadata={
                                "format": "pdf",
                                "page_number": index,
                                "parser": self.name,
                                "parser_quality": "blocks",
                                "reading_order": "pymupdf_blocks",
                                "blocks": text_blocks,
                                "has_tables": False,
                                "ocr_used": False,
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
