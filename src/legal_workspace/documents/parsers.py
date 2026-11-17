"""Conservative extraction: locators refer to extracted text, not rendered line positions."""

import io
import zipfile
from dataclasses import dataclass, field

from docx import Document as DocxDocument
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph
from pypdf import PdfReader

PARSER_VERSION = "text-v1:pypdf6:docx1"
MAX_TEXT = 2_000_000
MAX_PAGES = 1000


class DocumentError(ValueError):
    """A safe error code, never document contents."""


@dataclass
class Extraction:
    spans: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    status: str = "ready"
    length: int = 0

    def add(self, text: str, locator: dict):
        if self.length + len(text) > MAX_TEXT:
            raise DocumentError("text_limit")
        if not text.strip():
            return
        self.spans.append(
            {
                "ordinal": len(self.spans),
                "text": text,
                "locator": locator,
                "start": self.length,
                "end": self.length + len(text),
            }
        )
        self.length += len(text) + 1  # The canonical full text joins spans with a newline.


def extract(data: bytes, suffix: str) -> Extraction:
    result = Extraction()
    try:
        if suffix == ".txt":
            try:
                text = data.decode("utf-8-sig")
            except UnicodeDecodeError as exc:
                raise DocumentError("utf8_required") from exc
            if "\x00" in text:
                raise DocumentError("binary_text")
            for number, line in enumerate(text.splitlines(), 1):
                result.add(line, {"type": "line", "line": number})
        elif suffix == ".pdf":
            if not data.startswith(b"%PDF-"):
                raise DocumentError("invalid_pdf")
            reader = PdfReader(io.BytesIO(data), strict=True)
            if reader.is_encrypted:
                raise DocumentError("encrypted_pdf")
            if len(reader.pages) > MAX_PAGES:
                raise DocumentError("page_limit")
            for number, page in enumerate(reader.pages, 1):
                text = page.extract_text() or ""
                if not text.strip():
                    result.warnings.append(f"page_without_text:{number}")
                result.add(text, {"type": "page", "page": number})
            if result.warnings:
                result.status = "needs_review" if result.spans else "needs_ocr"
        elif suffix == ".docx":
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                if (
                    len(archive.infolist()) > 2000
                    or sum(i.file_size for i in archive.infolist()) > 50_000_000
                ):
                    raise DocumentError("docx_expansion_limit")
            document = DocxDocument(io.BytesIO(data))
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
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
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
        raise DocumentError("invalid_document") from exc
    if not result.spans and result.status != "needs_ocr":
        raise DocumentError("empty_document")
    return result
