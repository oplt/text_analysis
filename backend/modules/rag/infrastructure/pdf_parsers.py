"""PDF parsing adapters: pypdf baseline + PyMuPDF block/reading-order extraction."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Protocol

from backend.core.config import settings
from backend.modules.rag.domain.models import ParsedDocument

logger = logging.getLogger(__name__)

SUPPORTED_PDF_PARSERS = frozenset({"auto", "pymupdf", "pypdf"})


def _stable_block_id(*, page_number: int, block_index: int, text: str, bbox) -> str:
    payload = json.dumps(
        {
            "page_number": page_number,
            "block_index": block_index,
            "text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "bbox": list(bbox) if bbox else None,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"block-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:32]}"


class PdfParserAdapter(Protocol):
    def parse(self, content: bytes) -> list[ParsedDocument]: ...


class BasicPypdfParser:
    """Baseline PDF parser using pypdf page text extraction."""

    name = "pypdf-v1"

    def parse(
        self,
        content: bytes,
        *,
        configured_parser: str = "pypdf",
        fallback_reason: str | None = None,
    ) -> list[ParsedDocument]:
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
                            "parser_quality": "basic_text",
                            "configured_parser": configured_parser,
                            "fallback_used": fallback_reason is not None,
                            "fallback_reason": fallback_reason,
                            "layout_extraction_available": False,
                            "layout_extraction_ran": False,
                            "table_extraction_available": False,
                            "table_extraction_ran": False,
                            "ocr_ran": False,
                        },
                        page_number=index,
                    )
                )
        return docs


class EnhancedPdfParser:
    """Layout-aware parser with block/bbox reading-order metadata when pymupdf is available."""

    name = "pymupdf-blocks-v1"

    def parse(
        self,
        content: bytes,
        *,
        configured_parser: str = "pymupdf",
        fallback_reason: str | None = None,
    ) -> list[ParsedDocument]:
        try:
            import pymupdf as fitz
        except ImportError:
            try:
                import fitz  # type: ignore[no-redef]  # compatibility with older PyMuPDF
            except ImportError as exc:
                raise RuntimeError("enhanced PDF parser unavailable") from exc
        except Exception as exc:
            raise RuntimeError("enhanced PDF parser unavailable") from exc

        docs: list[ParsedDocument] = []
        with fitz.open(stream=content, filetype="pdf") as document:
            for index, page in enumerate(document, start=1):
                blocks = page.get_text("dict").get("blocks") or []
                text_blocks: list[dict] = []
                lines_out: list[str] = []
                for block_index, block in enumerate(blocks):
                    if block.get("type") != 0:
                        continue
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
                            "block_id": _stable_block_id(
                                page_number=index,
                                block_index=block_index,
                                text=block_text,
                                bbox=bbox,
                            ),
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
                                "parser_quality": (
                                    "enhanced_layout" if text_blocks else "enhanced_text"
                                ),
                                "configured_parser": configured_parser,
                                "fallback_used": fallback_reason is not None,
                                "fallback_reason": fallback_reason,
                                "layout_extraction_available": True,
                                "layout_extraction_ran": True,
                                "table_extraction_available": False,
                                "table_extraction_ran": False,
                                "ocr_ran": False,
                                "reading_order": "pymupdf_blocks",
                                "blocks": text_blocks,
                            },
                            page_number=index,
                        )
                    )
        if not docs:
            raise RuntimeError("enhanced PDF parser returned no text")
        return docs


def _configured_parser(parser_mode: str | None) -> str:
    configured = (parser_mode or settings.RAG_PDF_PARSER).strip().lower()
    if configured not in SUPPORTED_PDF_PARSERS:
        supported = ", ".join(sorted(SUPPORTED_PDF_PARSERS))
        raise ValueError(
            f"RAG_PDF_PARSER={configured!r} is not supported. Supported parsers: {supported}."
        )
    return configured


def parse_pdf_with_fallback(
    content: bytes, *, parser_mode: str | None = None
) -> list[ParsedDocument]:
    """Use the configured PDF parser and retain a reliable pypdf fallback."""
    configured = _configured_parser(parser_mode)
    if configured == "pypdf":
        return BasicPypdfParser().parse(content, configured_parser=configured)

    try:
        return EnhancedPdfParser().parse(content, configured_parser=configured)
    except Exception as exc:  # noqa: BLE001 - fallback is the parser contract
        logger.info("Enhanced PDF parser unavailable or failed; using pypdf baseline")
        return BasicPypdfParser().parse(
            content,
            configured_parser=configured,
            fallback_reason=f"{type(exc).__name__}: {exc}",
        )
