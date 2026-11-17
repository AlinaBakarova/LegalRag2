"""Limits use decimal MB, consistently in import, HTTP upload and the UI."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentLimits:
    file_bytes: int
    pdf_pages: int
    docx_expanded_bytes: int


USER_LIMITS = DocumentLimits(25_000_000, 300, 100_000_000)
BASE_LIMITS = DocumentLimits(300_000_000, 1000, 600_000_000)


def limits_for(kind: str) -> DocumentLimits:
    return BASE_LIMITS if kind == "base" else USER_LIMITS
