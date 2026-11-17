import hashlib
import io
import json
import logging
import os
import re
import tempfile
import time
from itertools import islice
from pathlib import Path
from uuid import uuid4

from filelock import FileLock
from sqlalchemy import delete, select, update

from legal_workspace.documents.limits import USER_LIMITS, limits_for
from legal_workspace.documents.parsers import PARSER_VERSION, DocumentError, extract
from legal_workspace.jobs.queue import Claim
from legal_workspace.persistence.database import Database
from legal_workspace.persistence.models import Document, IndexGeneration, Job, SourceSpan

MAX_FILE = USER_LIMITS.file_bytes
logger = logging.getLogger(__name__)


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

    @staticmethod
    def _enqueue(connection, document_id):
        now = time.time()
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

    def upload(self, filename: str, data: bytes, *, source=None, corpus_key=None) -> str:
        return self.upload_stream(filename, io.BytesIO(data), source=source, corpus_key=corpus_key)

    def upload_stream(self, filename, stream, *, source=None, corpus_key=None, expected_sha=None):
        filename = filename.replace("\\", "/").split("/")[-1][:255]
        if Path(filename).suffix.lower() not in {".txt", ".pdf", ".docx"}:
            raise DocumentError("unsupported_format")
        limits = limits_for("base" if corpus_key else "user")
        document_id = uuid4().hex
        path = self.path(document_id)
        temporary = path.with_suffix(".tmp")
        with self.lock:
            try:
                digest = hashlib.sha256()
                size = 0
                with temporary.open("xb") as output:
                    while chunk := stream.read(1024 * 1024):
                        size += len(chunk)
                        if size > limits.file_bytes:
                            raise DocumentError("file_size_limit")
                        digest.update(chunk)
                        output.write(chunk)
                    if not size:
                        raise DocumentError("file_size_limit")
                    output.flush()
                    os.fsync(output.fileno())
                checksum = digest.hexdigest()
                if expected_sha and checksum != expected_sha:
                    raise DocumentError("corpus_checksum_mismatch")
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
                        if existing["sha256"] != checksum or existing["source"] != source:
                            raise DocumentError("corpus_key_changed")
                        return existing["id"]
                temporary.replace(path)
                with self.db.transaction() as connection:
                    connection.execute(
                        Document.__table__.insert().values(
                            id=document_id,
                            kind="base" if corpus_key else "user",
                            filename=filename,
                            sha256=checksum,
                            size=size,
                            status="uploaded",
                            created_at=time.time(),
                            warnings=[],
                            source=source or {},
                            corpus_key=corpus_key,
                        )
                    )
                    self._enqueue(connection, document_id)
            except BaseException:
                with self.db.engine.connect() as connection:
                    exists = connection.scalar(
                        select(Document.id).where(Document.id == document_id)
                    )
                if not exists:
                    path.unlink(missing_ok=True)
                raise
            finally:
                temporary.unlink(missing_ok=True)
        return document_id

    def list(self):
        self._sync_failed_jobs()
        with self.db.engine.connect() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    select(Document.__table__)
                    .where(Document.status.not_in(["deleting", "deleted"]))
                    .order_by(Document.created_at.desc())
                ).mappings()
            ]

    def _sync_failed_jobs(self):
        # Exhausted retries after an infrastructure failure must not leave a spinner forever.
        with self.db.transaction() as connection:
            failed = (
                connection.execute(
                    select(Job.payload).where(
                        Job.kind == "document.extract", Job.status == "failed"
                    )
                )
                .scalars()
                .all()
            )
            for payload in failed:
                document_id = payload.get("document_id")
                connection.execute(
                    update(Document)
                    .where(Document.id == document_id, Document.status == "uploaded")
                    .values(status="failed", warnings=["processing_failed"])
                )

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
            try:
                original = self.path(document_id).open("rb")
            except FileNotFoundError:
                original = None
        # Anonymous spool keeps the full extracted corpus off the Python heap and is
        # removed by the OS even if the worker is killed. Publish all spans atomically.
        with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as spool:

            def emit(span):
                spool.write(json.dumps(span, ensure_ascii=False) + "\n")

            try:
                if original is None:
                    raise DocumentError("original_missing")
                with original:
                    digest = hashlib.file_digest(original, "sha256").hexdigest()
                    if digest != document["sha256"]:
                        raise DocumentError("original_checksum_mismatch")
                    original.seek(0)
                    extraction = extract(
                        original,
                        Path(document["filename"]).suffix.lower(),
                        limits=limits_for(document["kind"]),
                        sink=emit,
                    )
                status, warnings = extraction.status, extraction.warnings
            except DocumentError as exc:
                status, warnings = "failed", [str(exc)]
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
                self._invalidate_index(connection, document_id)
                connection.execute(delete(SourceSpan).where(SourceSpan.document_id == document_id))
                if status != "failed":
                    spool.seek(0)
                    rows = (dict(json.loads(line), document_id=document_id) for line in spool)
                    while batch := list(islice(rows, 50)):
                        connection.execute(SourceSpan.__table__.insert(), batch)
                connection.execute(
                    update(Document)
                    .where(Document.id == document_id)
                    .values(
                        status=status,
                        warnings=warnings,
                        parser_version=PARSER_VERSION,
                        text_revision=uuid4().hex,
                    )
                )
                if status in {"ready", "needs_review"}:
                    from legal_workspace.retrieval.service import SearchService

                    SearchService.enqueue(connection, document_id)
        logger.log(
            logging.WARNING if warnings else logging.INFO,
            "document_extracted document_id=%s job_id=%s status=%s warnings=%s",
            document_id,
            claim.id,
            status,
            warnings,
        )
        return {"document_id": document_id, "status": status}

    def reprocess(self, document_id):
        with self.lock:
            self.get(document_id)
            with self.db.transaction(immediate=True) as connection:
                self._delete_jobs(connection, document_id)
                self._invalidate_index(connection, document_id)
                connection.execute(delete(SourceSpan).where(SourceSpan.document_id == document_id))
                connection.execute(
                    update(Document)
                    .where(Document.id == document_id)
                    .values(
                        status="uploaded",
                        warnings=[],
                        text_revision=None,
                    )
                )
                self._enqueue(connection, document_id)

    @staticmethod
    def _delete_jobs(connection, document_id):
        rows = connection.execute(
            select(Job.id, Job.payload).where(Job.kind.in_(["document.extract", "search.index"]))
        ).all()
        for job_id, payload in rows:
            if payload.get("document_id") == document_id:
                connection.execute(delete(Job).where(Job.id == job_id))

    @staticmethod
    def _invalidate_index(connection, document_id):
        connection.execute(
            update(Document).where(Document.id == document_id).values(active_index_id=None)
        )
        connection.execute(
            delete(IndexGeneration).where(IndexGeneration.document_id == document_id)
        )

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
            self._delete_jobs(connection, document_id)
            connection.execute(delete(Document).where(Document.id == document_id))

    def recover(self):
        with self.lock:
            with self.db.engine.connect() as connection:
                documents = list(connection.execute(select(Document.id, Document.status)))
            known = {row.id for row in documents}
            for row in documents:
                if row.status in {"deleting", "deleted"}:
                    self._finish_delete(row.id)
            for path in self.root.iterdir():
                if path.is_file() and (
                    path.suffix == ".tmp"
                    or (re.fullmatch(r"[0-9a-f]{32}", path.name) and path.name not in known)
                ):
                    path.unlink()
