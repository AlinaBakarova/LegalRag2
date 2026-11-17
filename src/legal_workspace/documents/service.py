import hashlib
import os
import re
import time
from pathlib import Path
from uuid import uuid4

from filelock import FileLock
from sqlalchemy import delete, select, update

from legal_workspace.documents.parsers import PARSER_VERSION, DocumentError, extract
from legal_workspace.jobs.queue import Claim
from legal_workspace.persistence.database import Database
from legal_workspace.persistence.models import Document, Job, SourceSpan

MAX_FILE = 20_000_000


class DocumentService:
    def __init__(self, database: Database):
        self.db = database
        self.root = database.settings.data_dir / "originals"
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.lock = FileLock(str(database.settings.data_dir / ".documents.lock"), timeout=30)

    def path(self, document_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{32}", document_id):
            raise DocumentError("not_found")
        return self.root / document_id

    def upload(self, filename: str, data: bytes, *, source=None, corpus_key=None) -> str:
        filename = filename.replace("\\", "/").split("/")[-1][:255]
        if Path(filename).suffix.lower() not in {".txt", ".pdf", ".docx"}:
            raise DocumentError("unsupported_format")
        if not data or len(data) > MAX_FILE:
            raise DocumentError("file_size_limit")
        digest = hashlib.sha256(data).hexdigest()
        with self.lock:
            if corpus_key:
                with self.db.engine.connect() as connection:
                    existing = (
                        connection.execute(
                            select(Document.__table__).where(Document.corpus_key == corpus_key)
                        )
                        .mappings()
                        .first()
                    )
                    if existing:
                        if existing["sha256"] != digest or existing["source"] != source:
                            raise DocumentError("corpus_key_changed")
                        return existing["id"]
            document_id = uuid4().hex
            path = self.path(document_id)
            temporary = path.with_suffix(".tmp")
            try:
                with temporary.open("xb") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                temporary.replace(path)
                now = time.time()
                with self.db.transaction() as connection:
                    connection.execute(
                        Document.__table__.insert().values(
                            id=document_id,
                            kind="base" if corpus_key else "user",
                            filename=filename,
                            sha256=digest,
                            size=len(data),
                            status="uploaded",
                            created_at=now,
                            warnings=[],
                            source=source or {},
                            corpus_key=corpus_key,
                        )
                    )
                    connection.execute(
                        Job.__table__.insert().values(
                            id=uuid4().hex,
                            kind="document.extract",
                            status="queued",
                            payload={"document_id": document_id},
                            priority=0,
                            attempts=0,
                            max_attempts=3,
                            created_at=now,
                            updated_at=now,
                            available_at=now,
                        )
                    )
            except BaseException:
                temporary.unlink(missing_ok=True)
                # A crash after file commit is repaired by recover().
                with self.db.engine.connect() as connection:
                    exists = connection.scalar(
                        select(Document.id).where(Document.id == document_id)
                    )
                if not exists:
                    path.unlink(missing_ok=True)
                raise
            return document_id

    def list(self):
        with self.db.engine.connect() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    select(Document.__table__)
                    .where(Document.status.not_in(["deleting", "deleted"]))
                    .order_by(Document.created_at.desc())
                ).mappings()
            ]

    def get(self, document_id):
        self.path(document_id)
        with self.db.engine.connect() as connection:
            row = (
                connection.execute(
                    select(Document.__table__).where(
                        Document.id == document_id, Document.status.not_in(["deleting", "deleted"])
                    )
                )
                .mappings()
                .first()
            )
            if row is None:
                raise DocumentError("not_found")
            return dict(row)

    def spans(self, document_id):
        self.get(document_id)
        with self.db.engine.connect() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    select(SourceSpan.__table__)
                    .where(SourceSpan.document_id == document_id)
                    .order_by(SourceSpan.ordinal)
                ).mappings()
            ]

    def process(self, claim: Claim):
        document_id = claim.payload["document_id"]
        with self.lock:
            try:
                document = self.get(document_id)
            except DocumentError:
                return {"skipped": True}
            data = self.path(document_id).read_bytes()
        try:
            if hashlib.sha256(data).hexdigest() != document["sha256"]:
                raise DocumentError("original_checksum_mismatch")
            extraction = extract(data, Path(document["filename"]).suffix.lower())
            status, warnings, spans = extraction.status, extraction.warnings, extraction.spans
        except DocumentError as exc:
            status, warnings, spans = "failed", [str(exc)], []
        # Commit extracted text only while this exact lease is live and the document is available.
        with self.db.transaction(immediate=True) as connection:
            live = connection.scalar(
                select(Job.id).where(
                    Job.id == claim.id,
                    Job.status == "running",
                    Job.lease_token == claim.token,
                    Job.lease_expires_at > time.time(),
                )
            )
            active = connection.scalar(
                select(Document.id).where(
                    Document.id == document_id, Document.status.not_in(["deleting", "deleted"])
                )
            )
            if not live or not active:
                return {"skipped": True}
            connection.execute(delete(SourceSpan).where(SourceSpan.document_id == document_id))
            if spans:
                connection.execute(
                    SourceSpan.__table__.insert(), [dict(s, document_id=document_id) for s in spans]
                )
            connection.execute(
                update(Document)
                .where(Document.id == document_id)
                .values(
                    status=status,
                    warnings=warnings,
                    parser_version=PARSER_VERSION,
                )
            )
        return {"document_id": document_id, "status": status}

    def delete(self, document_id):
        with self.lock:
            document = self.get(document_id)
            if document["kind"] == "base":
                raise DocumentError("base_document_protected")
            with self.db.transaction() as connection:
                connection.execute(
                    update(Document).where(Document.id == document_id).values(status="deleting")
                )
            self._finish_delete(document_id)

    def _finish_delete(self, document_id):
        self.path(document_id).unlink(missing_ok=True)
        with self.db.transaction() as connection:
            connection.execute(delete(SourceSpan).where(SourceSpan.document_id == document_id))
            # Job payloads contain IDs only. No document text survives cleanup.
            rows = connection.execute(
                select(Job.id, Job.payload).where(Job.kind == "document.extract")
            ).all()
            for job_id, payload in rows:
                if payload.get("document_id") == document_id:
                    connection.execute(delete(Job).where(Job.id == job_id))
            connection.execute(
                update(Document)
                .where(Document.id == document_id)
                .values(
                    status="deleted",
                    filename="Удалённый документ",
                    source={},
                    warnings=[],
                )
            )

    def recover(self):
        with self.lock:
            with self.db.engine.connect() as connection:
                documents = list(connection.execute(select(Document.id, Document.status)))
            known = {row.id for row in documents}
            for row in documents:
                if row.status == "deleting":
                    self._finish_delete(row.id)
            for path in self.root.iterdir():
                if path.is_file() and (
                    path.suffix == ".tmp"
                    or (re.fullmatch(r"[0-9a-f]{32}", path.name) and path.name not in known)
                ):
                    path.unlink()
