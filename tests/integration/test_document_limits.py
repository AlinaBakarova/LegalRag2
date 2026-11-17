import io

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
from sqlalchemy import select

from legal_workspace.documents.limits import BASE_LIMITS, USER_LIMITS
from legal_workspace.documents.parsers import DocumentError, extract
from legal_workspace.documents.service import DocumentService
from legal_workspace.jobs.queue import JobQueue
from legal_workspace.jobs.worker import Worker
from legal_workspace.persistence.models import Document, Job, SourceSpan


def pdf_pages(count, text=b"x"):
    writer = PdfWriter()
    font = writer._add_object(
        DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
    )
    content = DecodedStreamObject()
    content.set_data(b"BT /F1 10 Tf 10 100 Td (" + text + b") Tj ET")
    content_id = writer._add_object(content)
    for _ in range(count):
        page = writer.add_blank_page(width=300, height=300)
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
        )
        page[NameObject("/Contents")] = content_id
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def test_pdf_page_boundaries_and_untruncated_base(database, settings):
    data = pdf_pages(1000, b"a" * 2100)
    with pytest.raises(DocumentError, match="page_limit"):
        extract(data, ".pdf", limits=USER_LIMITS)
    service = DocumentService(database)
    doc_id = service.upload("large.pdf", data, source={}, corpus_key="synthetic-large")
    Worker(database, settings).run_once()
    assert service.get(doc_id)["status"] == "ready"
    spans = service.spans(doc_id)
    assert len(spans) == 1000
    assert sum(len(s["text"]) for s in spans) > 2_000_000
    assert spans[-1]["locator"]["page"] == 1000
    assert spans[-1]["text"] == "a" * 2100
    assert len(extract(pdf_pages(300), ".pdf", limits=USER_LIMITS).spans) == 300
    with pytest.raises(DocumentError, match="page_limit"):
        extract(pdf_pages(301), ".pdf", limits=USER_LIMITS)
    with pytest.raises(DocumentError, match="page_limit"):
        extract(pdf_pages(1001), ".pdf", limits=BASE_LIMITS)


def test_streamed_size_boundaries(database, settings, tmp_path):
    service = DocumentService(database)
    file = tmp_path / "large.pdf"
    # A valid PDF with a large unused comment: exercises a 300 MB original without
    # constructing a 300 MB Python bytes object or inventing legal source material.
    prefix, trailer = pdf_pages(1).rsplit(b"startxref", 1)
    ending = b"\nstartxref" + trailer
    with file.open("wb") as stream:
        stream.write(prefix + b"%")
        stream.seek(BASE_LIMITS.file_bytes - len(ending))
        stream.write(ending)
    with file.open("rb") as stream:
        doc_id = service.upload_stream("large.pdf", stream, corpus_key="size-test", source={})
    assert service.get(doc_id)["size"] == 300_000_000
    assert service.path(doc_id).stat().st_size == 300_000_000
    Worker(database, settings).run_once()
    assert service.get(doc_id)["status"] == "ready"
    assert service.spans(doc_id)[0]["text"] == "x"
    # Failed upload cleans the temporary file and cannot leave a document or job.
    with file.open("rb") as stream:
        with pytest.raises(DocumentError, match="file_size_limit"):
            service.upload_stream("user.pdf", stream)
    assert len(service.list()) == 1
    assert list(service.root.glob("*.tmp")) == []
    with file.open("ab") as stream:
        stream.write(b"x")
    with file.open("rb") as stream:
        with pytest.raises(DocumentError, match="file_size_limit"):
            service.upload_stream("large.pdf", stream, corpus_key="too-large", source={})


def test_reprocess_existing_limit_error_without_reset(database, settings):
    from sqlalchemy import update

    service = DocumentService(database)
    doc_id = service.upload("legal.txt", b"Complete document", source={}, corpus_key="base")
    with database.transaction() as connection:
        connection.execute(
            update(Document)
            .where(Document.id == doc_id)
            .values(status="failed", warnings=["text_limit"])
        )
    service.reprocess(doc_id)
    Worker(database, settings).run_once()
    assert service.get(doc_id)["status"] == "ready"
    assert service.spans(doc_id)[0]["text"] == "Complete document"
    service.reprocess(doc_id)
    Worker(database, settings).run_once()
    assert len(service.spans(doc_id)) == 1


def test_hard_delete_removes_all_rows_and_jobs(database, settings):
    service = DocumentService(database)
    doc_id = service.upload("user.txt", b"content")
    Worker(database, settings).run_once()
    service.reprocess(doc_id)
    claim = JobQueue(database).claim("worker", 30)
    service.delete(doc_id)
    assert service.process(claim) == {"skipped": True}
    assert not service.path(doc_id).exists()
    with database.engine.connect() as connection:
        assert connection.scalar(select(Document.id)) is None
        assert connection.scalar(select(SourceSpan.id)) is None
        assert connection.scalar(select(Job.id)) is None


def test_user_exact_size_and_oversize_cleanup(database, tmp_path):
    service = DocumentService(database)
    file = tmp_path / "user.txt"
    with file.open("wb") as stream:
        stream.truncate(USER_LIMITS.file_bytes)
    with file.open("rb") as stream:
        doc_id = service.upload_stream("user.txt", stream)
    assert service.get(doc_id)["size"] == 25_000_000
    service.delete(doc_id)
    with file.open("ab") as stream:
        stream.write(b"x")
    with file.open("rb") as stream:
        with pytest.raises(DocumentError, match="file_size_limit"):
            service.upload_stream("user.txt", stream)
    assert service.list() == []
    assert list(service.root.iterdir()) == []


def test_formula_text_is_preserved_without_math_parser():
    from docx import Document as DocxDocument

    source = DocxDocument()
    formula = "Неустойка = сумма долга × ставка × число дней / 365"
    source.add_paragraph(formula)
    data = io.BytesIO()
    source.save(data)
    assert extract(data.getvalue(), ".docx").spans[0]["text"] == formula
    assert (
        extract(pdf_pages(1, b"penalty = debt * rate * days / 365"), ".pdf").spans[0]["text"]
        == "penalty = debt * rate * days / 365"
    )


def test_exhausted_infrastructure_failure_is_visible(database):
    service = DocumentService(database)
    doc_id = service.upload("user.txt", b"content")
    queue = JobQueue(database)
    claim = queue.claim("worker", 30)
    queue.fail(claim, "handler_failed", retryable=False)
    assert service.list()[0]["status"] == "failed"
    assert service.get(doc_id)["warnings"] == ["processing_failed"]


def test_unreadable_page_preserves_other_pages_and_warns(monkeypatch, caplog):
    from legal_workspace.documents import parsers

    class Page:
        def __init__(self, text):
            self.text = text

        def extract_text(self):
            if self.text is None:
                raise ValueError("PRIVATE_DOCUMENT_CONTENT_MUST_NOT_BE_LOGGED")
            return self.text

    class Reader:
        is_encrypted = False
        pages = [Page("page 1"), Page(None), Page("page 3")]

    monkeypatch.setattr(parsers, "PdfReader", lambda *args, **kwargs: Reader())
    parsed = extract(b"%PDF-test", ".pdf")
    assert parsed.status == "needs_review"
    assert parsed.warnings == ["page_read_failed:2"]
    assert [s["text"] for s in parsed.spans] == ["page 1", "page 3"]
    assert "PRIVATE_DOCUMENT_CONTENT" not in caplog.text
    assert "pdf_page_failed" in caplog.text
