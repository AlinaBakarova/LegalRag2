"""Conservative extraction: locators refer to extracted text, not rendered line positions."""

import io
import logging
import traceback
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import BinaryIO

from docx import Document as DocxDocument
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph
from pypdf import PdfReader

from legal_workspace.documents.limits import USER_LIMITS, DocumentLimits

PARSER_VERSION = "text-v2:pypdf6:docx1"
logger = logging.getLogger(__name__)


class DocumentError(ValueError):
    """A safe error code, never document contents."""


@dataclass
class Extraction:
    spans: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    status: str = "ready"
    length: int = 0
    count: int = 0
    sink: Callable[[dict], None] | None = None

    def add(self, text: str, locator: dict):
        if not text.strip():
            return
        span = {
            "ordinal": self.count,
            "text": text,
            "locator": locator,
            "start": self.length,
            "end": self.length + len(text),
        }
        if self.sink:
            self.sink(span)
        else:
            self.spans.append(span)
        self.count += 1
        self.length += len(text) + 1  # The canonical full text joins spans with a newline.


def extract(
    data: bytes | BinaryIO,
    suffix: str,
    *,
    limits: DocumentLimits = USER_LIMITS,
    sink: Callable[[dict], None] | None = None,
) -> Extraction:
    result = Extraction(sink=sink)
    stream = io.BytesIO(data) if isinstance(data, bytes) else data
    stream.seek(0)
    try:
        if suffix == ".txt":
            text_stream = io.TextIOWrapper(stream, encoding="utf-8-sig", newline=None)
            try:
                for number, line in enumerate(text_stream, 1):
                    if "\x00" in line:
                        raise DocumentError("binary_text")
                    result.add(line.rstrip("\r\n"), {"type": "line", "line": number})
            except UnicodeDecodeError as exc:
                raise DocumentError("utf8_required") from exc
            finally:
                text_stream.detach()
        elif suffix == ".pdf":
            if stream.read(5) != b"%PDF-":
                raise DocumentError("invalid_pdf")
            stream.seek(0)
            reader = PdfReader(stream, strict=True)
            if reader.is_encrypted:
                raise DocumentError("encrypted_pdf")
            if len(reader.pages) > limits.pdf_pages:
                raise DocumentError("page_limit")
            for number, page in enumerate(reader.pages, 1):
                try:
                    text = page.extract_text() or ""
                except Exception as exc:
                    frames = [
                        (f.filename, f.lineno, f.name)
                        for f in traceback.extract_tb(exc.__traceback__)
                    ]
                    logger.warning(
                        "pdf_page_failed page=%s error_type=%s stack=%s",
                        number,
                        type(exc).__name__,
                        frames,
                    )
                    result.warnings.append(f"page_read_failed:{number}")
                    continue
                if not text.strip():
                    result.warnings.append(f"page_without_text:{number}")
                result.add(text, {"type": "page", "page": number})
            if result.warnings:
                result.status = "needs_review" if result.count else "needs_ocr"
        elif suffix == ".docx":
            with zipfile.ZipFile(stream) as archive:
                if (
                    len(archive.infolist()) > 2000
                    or sum(i.file_size for i in archive.infolist()) > limits.docx_expanded_bytes
                ):
                    raise DocumentError("docx_expansion_limit")
            stream.seek(0)
            document = DocxDocument(stream)
            paragraph = table_number = 0
            for block in document.element.body:
                if block.tag == qn("w:p"):
                    paragraph += 1
                    result.add(
                        Paragraph(block, document).text,
                        {"type": "paragraph", "paragraph": paragraph},
                    )
                elif block.tag == qn("w:tbl"):
                    table_number += 1
                    table = Table(block, document)
                    for row_number, row in enumerate(table.rows, 1):
                        for cell_number, cell in enumerate(row.cells, 1):
                            result.add(
                                cell.text,
                                {
                                    "type": "cell",
                                    "table": table_number,
                                    "row": row_number,
                                    "cell": cell_number,
                                },
                            )
            # Never silently claim complete support for non-body content or tracked changes.
            xml = document.element.xml
            if any(
                token in xml
                for token in (
                    "<w:ins",
                    "<w:del",
                    "<w:drawing",
                    "<w:txbx",
                    "<w:sdt",
                    "<w:fldChar",
                    "<w:altChunk",
                    "<m:oMath",
                    "<w:object",
                )
            ):
                result.warnings.append("unsupported_docx_elements")
            with zipfile.ZipFile(stream) as archive:
                if any(
                    name.startswith(
                        (
                            "word/header",
                            "word/footer",
                            "word/footnotes",
                            "word/endnotes",
                            "word/comments",
                        )
                    )
                    for name in archive.namelist()
                ):
                    result.warnings.append("docx_additional_parts_not_extracted")
            if len(document.element.body.xpath(".//w:tbl")) > table_number:
                result.warnings.append("nested_tables_not_extracted")
            if result.warnings:
                result.status = "needs_review"
        else:
            raise DocumentError("unsupported_format")
    except DocumentError:
        raise
    except Exception as exc:
        # Stack locations help diagnosis without logging exception messages/document contents.
        frames = [(f.filename, f.lineno, f.name) for f in traceback.extract_tb(exc.__traceback__)]
        logger.error("document_parse_failed error_type=%s stack=%s", type(exc).__name__, frames)
        raise DocumentError("invalid_document") from exc
    if not result.count and result.status != "needs_ocr":
        raise DocumentError("empty_document")
    return result
