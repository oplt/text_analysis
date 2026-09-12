"""PDF parsing adapters: pypdf baseline + PyMuPDF block/reading-order extraction."""

from __future__ import annotations

import hashlib
import io
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


def _extract_ocr_text(page, fitz) -> tuple[str, dict]:
    """OCR a text-poor page only when explicitly enabled and dependencies exist."""
    if not settings.RAG_PDF_OCR_ENABLED:
        return "", {"ocr_enabled": False, "ocr_ran": False}
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return "", {
            "ocr_enabled": True,
            "ocr_available": False,
            "ocr_ran": False,
            "ocr_unavailable_reason": "pytesseract_or_pillow_not_installed",
        }
    try:
        pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        image = Image.open(io.BytesIO(pixmap.tobytes("png")))
        text = pytesseract.image_to_string(image).strip()
        return text, {
            "ocr_enabled": True,
            "ocr_available": True,
            "ocr_ran": bool(text),
            "ocr_engine": "tesseract",
            "ocr_engine_version": str(pytesseract.get_tesseract_version()),
            "ocr_transformed_extract": bool(text),
        }
    except Exception as exc:  # noqa: BLE001 - OCR is optional enrichment
        logger.info("Optional PDF OCR was unavailable: %s", type(exc).__name__)
        return "", {
            "ocr_enabled": True,
            "ocr_available": True,
            "ocr_ran": False,
            "ocr_unavailable_reason": type(exc).__name__,
        }


def _extract_tables(page, *, page_number: int) -> list[dict]:
    """Return table cells as structured blocks, never prose-flattened text."""
    if not settings.RAG_PDF_TABLE_EXTRACTION_ENABLED or not hasattr(page, "find_tables"):
        return []
    try:
        found = page.find_tables()
        tables = getattr(found, "tables", found) or []
        result: list[dict] = []
        for table_index, table in enumerate(tables):
            rows = table.extract()
            normalized_rows = [[cell if cell is not None else "" for cell in row] for row in rows]
            bbox = list(table.bbox) if getattr(table, "bbox", None) else None
            table_id = _stable_block_id(
                page_number=page_number,
                block_index=10_000 + table_index,
                text=json.dumps(normalized_rows, ensure_ascii=False),
                bbox=bbox,
            )
            result.append(
                {
                    "block_id": table_id,
                    "table_index": table_index,
                    "page_number": page_number,
                    "bbox": bbox,
                    "rows": normalized_rows,
                }
            )
        return result
    except Exception as exc:  # noqa: BLE001 - tables are optional enrichment
        logger.info("Optional PDF table extraction failed: %s", type(exc).__name__)
        return []


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
                            "source_span_ids": [f"pdf-page-{index}"],
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
                text_poor = len(text) < max(1, settings.RAG_PDF_OCR_MIN_TEXT_CHARS)
                ocr_text, ocr_metadata = (
                    _extract_ocr_text(page, fitz)
                    if text_poor
                    else ("", {"ocr_enabled": False, "ocr_ran": False})
                )
                if ocr_text:
                    text = ocr_text
                tables = _extract_tables(page, page_number=index)
                if text:
                    docs.append(
                        ParsedDocument(
                            content=text,
                            metadata={
                                "format": "pdf",
                                "page_number": index,
                                "source_span_ids": [
                                    f"pdf-page-{index}",
                                    *(block["block_id"] for block in text_blocks),
                                ],
                                "parser": self.name,
                                "parser_quality": (
                                    "enhanced_layout" if text_blocks else "enhanced_text"
                                ),
                                "configured_parser": configured_parser,
                                "fallback_used": fallback_reason is not None,
                                "fallback_reason": fallback_reason,
                                "layout_extraction_available": True,
                                "layout_extraction_ran": True,
                                "table_extraction_available": hasattr(page, "find_tables"),
                                "table_extraction_ran": bool(tables),
                                "table_blocks": tables,
                                "text_poor": text_poor,
                                **ocr_metadata,
                                "reading_order": "pymupdf_blocks",
                                "blocks": text_blocks,
                            },
                            page_number=index,
                        )
                    )
                for table in tables:
                    docs.append(
                        ParsedDocument(
                            content=json.dumps(
                                {
                                    "kind": "pdf_table",
                                    "table_index": table["table_index"],
                                    "rows": table["rows"],
                                },
                                ensure_ascii=False,
                            ),
                            metadata={
                                "format": "pdf",
                                "parsed_block_type": "table",
                                "page_number": index,
                                "source_span_ids": [table["block_id"]],
                                "parser": self.name,
                                "table_extraction_available": True,
                                "table_extraction_ran": True,
                                "table": table,
                                "ocr_ran": False,
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
