import hashlib
import io
import json
from pathlib import Path

import pytest
from docx import Document as DocxDocument
from fastapi.testclient import TestClient
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
from sqlalchemy import select, update

from legal_workspace.api import create_api
from legal_workspace.documents.corpus_import import import_corpus
from legal_workspace.documents.parsers import DocumentError, extract
from legal_workspace.documents.service import DocumentService
from legal_workspace.jobs.queue import JobQueue
from legal_workspace.jobs.worker import Worker
from legal_workspace.persistence.models import Document, SourceSpan


def pdf_bytes(mixed=False):
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})}
    )
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 20 200 Td (Synthetic test document) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream)
    if mixed:
        writer.add_blank_page(width=300, height=300)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def test_pdf_text_and_mixed_pages():
    parsed = extract(pdf_bytes(), ".pdf")
    assert "Synthetic test document" in parsed.spans[0]["text"]
    assert parsed.spans[0]["locator"] == {"type": "page", "page": 1}
    assert parsed.status == "ready"
    mixed = extract(pdf_bytes(mixed=True), ".pdf")
    assert mixed.status == "needs_review"
    assert mixed.warnings == ["page_without_text:2"]
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)
    buffer = io.BytesIO()
    writer.write(buffer)
    assert extract(buffer.getvalue(), ".pdf").status == "needs_ocr"


def test_docx_order_tables_and_unsupported_parts():
    document = DocxDocument()
    document.add_paragraph("Первый абзац")
    document.add_table(rows=1, cols=1).cell(0, 0).text = "Ячейка"
    document.add_paragraph("Последний абзац")
    document.sections[0].header.paragraphs[0].text = "Заголовок вне основного текста"
    output = io.BytesIO()
    document.save(output)
    parsed = extract(output.getvalue(), ".docx")
    assert [span["text"] for span in parsed.spans] == ["Первый абзац", "Ячейка", "Последний абзац"]
    assert parsed.spans[1]["locator"]["cell"] == 1
    assert parsed.status == "needs_review"


@pytest.mark.parametrize(
    "data,suffix,code",
    [
        (b"broken", ".pdf", "invalid_pdf"),
        (b"broken", ".docx", "invalid_document"),
        (b"\xff", ".txt", "utf8_required"),
        (b"\x00", ".txt", "binary_text"),
        (b"  ", ".txt", "empty_document"),
    ],
)
def test_bad_files(data, suffix, code):
    with pytest.raises(DocumentError, match=code):
        extract(data, suffix)


def test_upload_extract_persist_delete(database, settings):
    service = DocumentService(database)
    data = "Строка 1\n\nСтрока 3".encode()
    doc_id = service.upload("../../sample.txt", data)
    assert service.get(doc_id)["filename"] == "sample.txt"
    assert service.path(doc_id).read_bytes() == data
    Worker(database, settings).run_once()
    assert service.get(doc_id)["status"] == "ready"
    spans = service.spans(doc_id)
    assert [span["locator"]["line"] for span in spans] == [1, 3]
    full = "\n".join(span["text"] for span in spans)
    assert all(full[span["start"] : span["end"]] == span["text"] for span in spans)
    database.initialize()
    assert service.spans(doc_id) == spans
    service.delete(doc_id)
    assert service.list() == []
    assert not service.path(doc_id).exists()
    with database.engine.connect() as connection:
        assert connection.scalar(select(SourceSpan.id)) is None


def test_error_status_and_delete_before_processing(database, settings):
    service = DocumentService(database)
    bad = service.upload("bad.pdf", b"bad")
    Worker(database, settings).run_once()
    assert service.get(bad)["status"] == "failed"
    assert service.get(bad)["warnings"] == ["invalid_pdf"]
    pending = service.upload("pending.txt", b"pending")
    service.delete(pending)
    assert not Worker(database, settings).run_once()


def test_deleted_document_cannot_be_resurrected(database):
    service = DocumentService(database)
    doc_id = service.upload("pending.txt", b"pending")
    claim = JobQueue(database).claim("worker", 30)
    service.delete(doc_id)
    assert service.process(claim) == {"skipped": True}
    assert service.list() == []


def test_recovery_of_interrupted_delete_and_orphan(database):
    service = DocumentService(database)
    doc_id = service.upload("pending.txt", b"pending")
    with database.transaction() as connection:
        connection.execute(update(Document).where(Document.id == doc_id).values(status="deleting"))
    orphan = service.root / ("a" * 32)
    orphan.write_bytes(b"orphan")
    temporary = service.root / "interrupted.tmp"
    temporary.write_bytes(b"temp")
    database.initialize()
    assert not orphan.exists() and not temporary.exists()
    assert not service.path(doc_id).exists()
    assert service.list() == []


def test_corpus_idempotency_checksum_and_protection(database, tmp_path):
    service = DocumentService(database)
    data = b"SYNTHETIC TEST ONLY - NOT LEGISLATION"
    (tmp_path / "test.txt").write_bytes(data)
    manifest = tmp_path / "manifest.json"
    metadata = {
        "key": "synthetic",
        "file": "test.txt",
        "sha256": hashlib.sha256(data).hexdigest(),
        "title": "Synthetic test",
        "source_url": "https://example.org/synthetic",
        "edition": "TEST",
        "verified_on": "2026-11-17",
    }
    manifest.write_text(json.dumps({"documents": [metadata]}))
    first = import_corpus(service, manifest)
    assert import_corpus(service, manifest) == first
    assert len(service.list()) == 1
    with pytest.raises(DocumentError, match="base_document_protected"):
        service.delete(first[0])
    (tmp_path / "test.txt").write_bytes(b"changed")
    with pytest.raises(DocumentError, match="corpus_checksum_mismatch"):
        import_corpus(service, manifest)


def test_api_upload_original_and_delete(settings):
    with TestClient(create_api(settings), base_url="http://127.0.0.1") as client:
        response = client.post(
            "/api/documents", files={"file": ("test.txt", b"hello", "text/plain")}
        )
        assert response.status_code == 202
        doc_id = response.json()["id"]
        assert client.get(f"/api/documents/{doc_id}/original").content == b"hello"
        assert client.get(f"/api/documents/{doc_id}/spans").json() == []
        assert client.delete(f"/api/documents/{doc_id}").status_code == 204
        assert client.get(f"/api/documents/{doc_id}").status_code == 404
        assert (
            client.post("/api/documents", files={"file": ("test.exe", b"bad")}).status_code == 400
        )


def test_lost_lease_cannot_publish_extraction(database, monkeypatch):
    from legal_workspace.documents import service as service_module

    service = DocumentService(database)
    doc_id = service.upload("test.txt", b"content")
    claim = JobQueue(database).claim("worker", 30)
    original_extract = service_module.extract

    def expire(data, suffix):
        from legal_workspace.persistence.models import Job

        with database.transaction() as connection:
            connection.execute(update(Job).where(Job.id == claim.id).values(lease_expires_at=0))
        return original_extract(data, suffix)

    monkeypatch.setattr(service_module, "extract", expire)
    assert service.process(claim) == {"skipped": True}
    assert service.get(doc_id)["status"] == "uploaded"
    assert service.spans(doc_id) == []


def test_upgrade_stage_one_keeps_existing_jobs(settings):
    from alembic import command
    from alembic.config import Config

    from legal_workspace.persistence import database as database_module
    from legal_workspace.persistence.database import Database

    db = Database(settings)
    settings.data_dir.mkdir(parents=True)
    config = Config()
    config.set_main_option(
        "script_location", str(Path(database_module.__file__).parent / "migrations")
    )
    try:
        with db.engine.begin() as connection:
            config.attributes["connection"] = connection
            command.upgrade(config, "0001_foundation")
        job_id = JobQueue(db).enqueue("system.check")
        db.initialize()
        assert JobQueue(db).get(job_id)["status"] == "queued"
        assert DocumentService(db).list() == []
    finally:
        db.close()


def test_api_cannot_delete_base_document(settings):
    with TestClient(create_api(settings), base_url="http://127.0.0.1") as client:
        service = DocumentService(client.app.state.database)
        doc_id = service.upload(
            "base.txt", b"SYNTHETIC TEST", source={"title": "TEST"}, corpus_key="test"
        )
        assert client.delete(f"/api/documents/{doc_id}").status_code == 403
        assert client.get(f"/api/documents/{doc_id}/original").content == b"SYNTHETIC TEST"


@pytest.mark.parametrize("suffix", [".pdf", ".docx"])
def test_formats_through_api_and_worker(settings, suffix):
    if suffix == ".pdf":
        data = pdf_bytes()
    else:
        source = DocxDocument()
        source.add_paragraph("Тестовый абзац")
        output = io.BytesIO()
        source.save(output)
        data = output.getvalue()
    with TestClient(create_api(settings), base_url="http://127.0.0.1") as client:
        response = client.post("/api/documents", files={"file": ("test" + suffix, data)})
        assert response.status_code == 202
        doc_id = response.json()["id"]
        Worker(client.app.state.database, settings).run_once()
        assert client.get(f"/api/documents/{doc_id}").json()["status"] == "ready"
        assert client.get(f"/api/documents/{doc_id}/spans").json()
        assert client.get(f"/api/documents/{doc_id}/original").content == data


def test_modified_original_is_not_processed(database, settings):
    service = DocumentService(database)
    doc_id = service.upload("test.txt", b"original")
    service.path(doc_id).write_bytes(b"changed")
    Worker(database, settings).run_once()
    assert service.get(doc_id)["warnings"] == ["original_checksum_mismatch"]
    assert service.spans(doc_id) == []
