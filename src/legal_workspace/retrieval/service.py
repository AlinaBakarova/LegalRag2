import hashlib
import json
import logging
import time
from importlib.metadata import version
from itertools import islice
from uuid import uuid4

import numpy as np
from sqlalchemy import delete, func, select, text, update

from legal_workspace.jobs.queue import Claim
from legal_workspace.persistence.models import Chunk, Document, IndexGeneration, Job, SourceSpan
from legal_workspace.retrieval.chunking import CHUNKING_VERSION, chunk_span, normalize
from legal_workspace.retrieval.embeddings import SearchError, get_embedder

AVAILABLE = ("ready", "needs_review")
logger = logging.getLogger(__name__)


class SearchService:
    def __init__(self, database, embedder=None):
        self.db = database
        self._embedder = embedder

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = get_embedder(
                self.db.settings.data_dir, self.db.settings.embedding_threads
            )
        return self._embedder

    @property
    def fingerprint(self):
        config = {
            "embedding": self.embedder.identity,
            "dimension": self.embedder.dimension,
            "chunking": CHUNKING_VERSION,
            "normalization": "snowball-ru-v1:" + version("snowballstemmer"),
            "format": "float32-normalized-v1",
        }
        return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()

    @staticmethod
    def enqueue(connection, document_id):
        # Serialize callers with the same writer transaction to avoid duplicate jobs.
        pending = connection.execute(
            select(Job.payload).where(
                Job.kind == "search.index", Job.status.in_(["queued", "running"])
            )
        ).scalars()
        if any(payload.get("document_id") == document_id for payload in pending):
            return False
        for job_id, payload in connection.execute(
            select(Job.id, Job.payload).where(Job.kind == "search.index", Job.status == "failed")
        ).all():
            if payload.get("document_id") == document_id:
                connection.execute(delete(Job).where(Job.id == job_id))
        now = time.time()
        connection.execute(
            Job.__table__.insert().values(
                id=uuid4().hex,
                kind="search.index",
                payload={"document_id": document_id},
                status="queued",
                priority=-1,
                attempts=0,
                max_attempts=3,
                created_at=now,
                updated_at=now,
                available_at=now,
            )
        )
        return True

    def schedule_missing(self, force=False):
        fingerprint = self.fingerprint  # Fail early if the model hasn't been prepared.
        with self.db.transaction(immediate=True) as connection:
            documents = connection.execute(
                select(Document.id, Document.active_index_id).where(
                    Document.status.in_(AVAILABLE), Document.text_revision.is_not(None)
                )
            ).all()
            count = 0
            for document_id, active in documents:
                current = connection.scalar(
                    select(IndexGeneration.fingerprint).where(IndexGeneration.id == active)
                )
                if force or current != fingerprint:
                    count += self.enqueue(connection, document_id)
            return count

    def _live(self, connection, claim, revision):
        job = connection.scalar(
            select(Job.id).where(
                Job.id == claim.id,
                Job.lease_token == claim.token,
                Job.status == "running",
                Job.lease_expires_at > time.time(),
            )
        )
        document = connection.scalar(
            select(Document.id).where(
                Document.id == claim.payload["document_id"],
                Document.text_revision == revision,
                Document.status.in_(AVAILABLE),
            )
        )
        if not job or not document:
            raise SearchError("stale_index_job")

    def process(self, claim: Claim):
        document_id = claim.payload["document_id"]
        generation = uuid4().hex
        with self.db.engine.connect() as connection:
            document = connection.execute(
                select(Document.text_revision).where(
                    Document.id == document_id, Document.status.in_(AVAILABLE)
                )
            ).first()
        if document is None or not document.text_revision:
            return {"skipped": True}
        revision = document.text_revision
        # Load the real model outside any write transaction.
        fingerprint = self.fingerprint
        try:
            with self.db.transaction(immediate=True) as connection:
                self._live(connection, claim, revision)
                connection.execute(
                    delete(IndexGeneration).where(
                        IndexGeneration.document_id == document_id,
                        IndexGeneration.state == "building",
                        IndexGeneration.job_id == claim.id,
                    )
                )
                connection.execute(
                    IndexGeneration.__table__.insert().values(
                        id=generation,
                        document_id=document_id,
                        text_revision=revision,
                        fingerprint=fingerprint,
                        dimension=self.embedder.dimension,
                        state="building",
                        job_id=claim.id,
                        lease_token=claim.token,
                        created_at=time.time(),
                    )
                )
            last_span = -1
            ordinal = 0
            heading = ""
            while True:
                with self.db.engine.connect() as connection:
                    spans = list(
                        connection.execute(
                            select(SourceSpan.__table__)
                            .where(
                                SourceSpan.document_id == document_id,
                                SourceSpan.ordinal > last_span,
                            )
                            .order_by(SourceSpan.ordinal)
                            .limit(16)
                        ).mappings()
                    )
                if not spans:
                    break
                for span in spans:
                    pieces = chunk_span(dict(span), heading, self.embedder)
                    while batch := list(islice(pieces, 16)):
                        heading = batch[-1].heading
                        vectors = self.embedder.embed(
                            [piece.heading + "\n" + piece.text for piece in batch]
                        )
                        if (
                            vectors.shape != (len(batch), self.embedder.dimension)
                            or not np.isfinite(vectors).all()
                        ):
                            raise SearchError("invalid_embeddings")
                        norms = np.linalg.norm(vectors, axis=1)
                        if np.any(norms == 0):
                            raise SearchError("invalid_embeddings")
                        rows = []
                        for piece, vector, norm in zip(batch, vectors, norms, strict=True):
                            rows.append(
                                dict(
                                    generation_id=generation,
                                    span_id=piece.span_id,
                                    ordinal=ordinal,
                                    start=piece.start,
                                    end=piece.end,
                                    text=piece.text,
                                    heading=piece.heading,
                                    terms=normalize(piece.heading + " " + piece.text),
                                    embedding=np.asarray(vector / norm, dtype="<f4").tobytes(),
                                )
                            )
                            ordinal += 1
                        with self.db.transaction(immediate=True) as connection:
                            self._live(connection, claim, revision)
                            connection.execute(Chunk.__table__.insert(), rows)
                    last_span = span["ordinal"]
            with self.db.transaction(immediate=True) as connection:
                self._live(connection, claim, revision)
                count = connection.scalar(
                    select(func.count()).select_from(Chunk).where(Chunk.generation_id == generation)
                )
                if not ordinal or count != ordinal:
                    raise SearchError("incomplete_index")
                connection.execute(
                    update(IndexGeneration)
                    .where(IndexGeneration.id == generation)
                    .values(state="ready")
                )
                connection.execute(
                    update(Document)
                    .where(Document.id == document_id)
                    .values(active_index_id=generation)
                )
                connection.execute(
                    delete(IndexGeneration).where(
                        IndexGeneration.document_id == document_id, IndexGeneration.id != generation
                    )
                )
            logger.info(
                "index_built document_id=%s generation_id=%s chunks=%s",
                document_id,
                generation,
                ordinal,
            )
            return {"indexed": True, "chunks": ordinal}
        except (Exception, KeyboardInterrupt):
            with self.db.transaction() as connection:
                connection.execute(
                    delete(IndexGeneration).where(
                        IndexGeneration.id == generation, IndexGeneration.state == "building"
                    )
                )
            raise

    def recover(self):
        # Remove only abandoned builds: never remove another live worker's staging data.
        with self.db.transaction(immediate=True) as connection:
            builds = connection.execute(
                select(
                    IndexGeneration.id, IndexGeneration.job_id, IndexGeneration.lease_token
                ).where(IndexGeneration.state == "building")
            ).all()
            for build in builds:
                live = connection.scalar(
                    select(Job.id).where(
                        Job.id == build.job_id,
                        Job.status == "running",
                        Job.lease_token == build.lease_token,
                        Job.lease_expires_at > time.time(),
                    )
                )
                if not live:
                    connection.execute(
                        delete(IndexGeneration).where(IndexGeneration.id == build.id)
                    )

    def status(self):
        with self.db.engine.connect() as connection:
            available = connection.scalar(
                select(func.count()).select_from(Document).where(Document.status.in_(AVAILABLE))
            )
        try:
            _ = self.fingerprint
        except SearchError:
            return {"model_ready": False, "available": available, "indexed": 0, "failed": 0}
        with self.db.engine.connect() as connection:
            indexed_ids = set(
                connection.execute(
                    select(Document.id)
                    .join(IndexGeneration, IndexGeneration.id == Document.active_index_id)
                    .where(*self._eligible())
                ).scalars()
            )
            failed_ids = {
                payload["document_id"]
                for payload in connection.execute(
                    select(Job.payload).where(Job.kind == "search.index", Job.status == "failed")
                ).scalars()
            }
            eligible_ids = set(
                connection.execute(
                    select(Document.id).where(Document.status.in_(AVAILABLE))
                ).scalars()
            )
        return {
            "model_ready": True,
            "available": available,
            "indexed": len(indexed_ids),
            "failed": len((failed_ids & eligible_ids) - indexed_ids),
        }

    def _eligible(self):
        return (
            Document.status.in_(AVAILABLE),
            IndexGeneration.id == Document.active_index_id,
            IndexGeneration.state == "ready",
            IndexGeneration.fingerprint == self.fingerprint,
            IndexGeneration.text_revision == Document.text_revision,
        )

    def search(self, query: str, *, mode="hybrid", document_ids=None, limit=10):
        if mode not in {"hybrid", "lexical", "semantic"} or not query.strip() or len(query) > 1500:
            raise SearchError("invalid_query")
        terms = normalize(query).split()[:50]
        if not terms:
            return []
        fingerprint = self.fingerprint
        if self.embedder.token_count(query) > self.embedder.max_tokens:
            raise SearchError("query_too_long")
        allowed = set(document_ids) if document_ids is not None else None
        with self.db.engine.connect() as connection:
            candidates = select(Document.id).where(Document.status.in_(["uploaded", *AVAILABLE]))
            active = (
                select(Document.id)
                .join(IndexGeneration, IndexGeneration.id == Document.active_index_id)
                .where(*self._eligible())
            )
            if allowed is not None:
                candidates = candidates.where(Document.id.in_(allowed))
                active = active.where(Document.id.in_(allowed))
            has_candidates = connection.execute(candidates.limit(1)).first() is not None
            has_active = connection.execute(active.limit(1)).first() is not None
        if has_candidates and not has_active:
            raise SearchError("index_not_ready")
        # Keep memory bounded by scanning vectors in batches, rather than caching the whole corpus.
        rankings = []
        with self.db.engine.connect() as connection:
            if mode in {"hybrid", "lexical"}:
                fts_query = " OR ".join(
                    '"' + term.replace('"', '""') + '"' for term in dict.fromkeys(terms)
                )
                sql = text("""
                    SELECT c.id, d.id AS document_id, bm25(chunk_fts) AS rank
                    FROM chunk_fts JOIN chunks c ON c.id=chunk_fts.rowid
                    JOIN index_generations g ON g.id=c.generation_id
                    JOIN documents d ON d.id=g.document_id
                    WHERE chunk_fts MATCH :query AND d.status IN ('ready','needs_review')
                    AND d.active_index_id=g.id AND g.state='ready'
                    AND g.text_revision=d.text_revision AND g.fingerprint=:fingerprint
                    ORDER BY rank, c.id
                """)
                rows = connection.execute(sql, {"query": fts_query, "fingerprint": fingerprint})
                lexical = []
                for row in rows:
                    if allowed is None or row.document_id in allowed:
                        lexical.append(row.id)
                        if len(lexical) == 60:
                            break
                rankings.append(lexical)
            if mode in {"hybrid", "semantic"}:
                vectors = self.embedder.embed([query])
                if vectors.shape != (1, self.embedder.dimension) or not np.isfinite(vectors).all():
                    raise SearchError("invalid_embeddings")
                vector = vectors[0]
                norm = np.linalg.norm(vector)
                if not norm:
                    raise SearchError("invalid_embeddings")
                vector = vector / norm
                scores = []
                statement = (
                    select(Chunk.id, Chunk.embedding)
                    .join(IndexGeneration, IndexGeneration.id == Chunk.generation_id)
                    .join(Document, Document.id == IndexGeneration.document_id)
                    .where(*self._eligible())
                )
                if allowed is not None:
                    statement = statement.where(Document.id.in_(allowed))
                result = connection.execute(statement).yield_per(256)
                for batch in result.partitions(256):
                    if any(len(row.embedding) != self.embedder.dimension * 4 for row in batch):
                        raise SearchError("incompatible_index")
                    matrix = np.stack([np.frombuffer(row.embedding, dtype="<f4") for row in batch])
                    if matrix.shape[1] != self.embedder.dimension or not np.isfinite(matrix).all():
                        raise SearchError("incompatible_index")
                    scores.extend(
                        (float(score), row.id)
                        for score, row in zip(matrix @ vector, batch, strict=True)
                    )
                    scores = sorted(scores, key=lambda item: (-item[0], item[1]))[:60]
                rankings.append([chunk_id for _score, chunk_id in scores])
        fused = {}
        for ranking in rankings:
            for rank, chunk_id in enumerate(ranking, 1):
                fused[chunk_id] = fused.get(chunk_id, 0) + 1 / (60 + rank)
        selected = sorted(fused, key=lambda chunk_id: (-fused[chunk_id], chunk_id))[:limit]
        results = []
        for chunk_id in selected:
            try:
                item = self.source(chunk_id)
            except SearchError:
                continue  # A source removed while searching cannot be returned.
            item["score"] = fused[chunk_id]
            results.append(item)
        return results

    def source(self, chunk_id):
        with self.db.engine.connect() as connection:
            row = (
                connection.execute(
                    select(
                        Chunk.id,
                        Chunk.text,
                        Chunk.heading,
                        Chunk.start,
                        Chunk.end,
                        SourceSpan.text.label("source_text"),
                        SourceSpan.locator,
                        Document.id.label("document_id"),
                        Document.filename,
                        Document.source,
                        Document.warnings,
                        Document.sha256,
                    )
                    .join(SourceSpan, SourceSpan.id == Chunk.span_id)
                    .join(IndexGeneration, IndexGeneration.id == Chunk.generation_id)
                    .join(Document, Document.id == IndexGeneration.document_id)
                    .where(Chunk.id == chunk_id, *self._eligible())
                )
                .mappings()
                .first()
            )
            if row is None:
                raise SearchError("source_not_found")
            if row["source_text"][row["start"] : row["end"]] != row["text"]:
                raise SearchError("source_mismatch")
            item = dict(row)
            item["title"] = item["source"].get("title", item["filename"])
            item.pop("source_text")  # Return the verified result excerpt, never a whole document.
            return item
