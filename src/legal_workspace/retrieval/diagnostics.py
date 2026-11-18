"""Read-only comparison against the legacy rankings on the existing index."""

import sqlite3
from types import SimpleNamespace

import numpy as np
from sqlalchemy import create_engine, select, text

from legal_workspace.persistence.models import Chunk, Document, IndexGeneration
from legal_workspace.rag.service import ChatService
from legal_workspace.retrieval.chunking import normalize
from legal_workspace.retrieval.service import SearchService


def diagnose(settings, question, chunk_id, *, embedder=None):
    if not question.strip() or len(question) > 1500:
        raise ValueError("Invalid diagnostic question")
    uri = settings.database_path.as_uri() + "?mode=ro"
    engine = create_engine("sqlite://", creator=lambda: sqlite3.connect(uri, uri=True))
    database = SimpleNamespace(settings=settings, engine=engine)
    search = SearchService(database, embedder)
    try:
        fingerprint = search.fingerprint
        terms = list(dict.fromkeys(normalize(question).split()[:50]))
        fts_query = " OR ".join('"' + term + '"' for term in terms)
        with engine.connect() as connection:
            lexical = (
                connection.execute(
                    text("""
                SELECT c.id, bm25(chunk_fts) AS score
                FROM chunk_fts JOIN chunks c ON c.id=chunk_fts.rowid
                JOIN index_generations g ON g.id=c.generation_id
                JOIN documents d ON d.id=g.document_id
                WHERE chunk_fts MATCH :query AND d.status IN ('ready','needs_review')
                AND d.active_index_id=g.id AND g.state='ready'
                AND g.text_revision=d.text_revision AND g.fingerprint=:fingerprint
                ORDER BY score,c.id
            """),
                    {"query": fts_query, "fingerprint": fingerprint},
                ).all()
                if terms
                else []
            )
            vector = search.embedder.embed([question])[0]
            vector = vector / np.linalg.norm(vector)
            semantic = []
            rows = connection.execute(
                select(Chunk.id, Chunk.embedding)
                .join(IndexGeneration, IndexGeneration.id == Chunk.generation_id)
                .join(Document, Document.id == IndexGeneration.document_id)
                .where(*search._eligible())
            ).yield_per(256)
            for batch in rows.partitions(256):
                matrix = np.stack([np.frombuffer(r.embedding, dtype="<f4") for r in batch])
                semantic.extend(
                    (r.id, float(score)) for r, score in zip(batch, matrix @ vector, strict=True)
                )
        semantic.sort(key=lambda pair: (-pair[1], pair[0]))
        lexical_positions = {r.id: (rank, r.score) for rank, r in enumerate(lexical, 1)}
        vector_positions = {cid: (rank, score) for rank, (cid, score) in enumerate(semantic, 1)}
        fused = {}
        for ranking in (lexical[:60], semantic[:60]):
            for rank, pair in enumerate(ranking, 1):
                fused[pair[0]] = fused.get(pair[0], 0) + 1 / (60 + rank)
        fused_positions = {
            cid: rank for rank, cid in enumerate(sorted(fused, key=lambda c: (-fused[c], c)), 1)
        }

        def details(cid):
            fts_rank, bm25 = lexical_positions.get(cid, (None, None))
            vector_rank, cosine = vector_positions.get(cid, (None, None))
            return {
                "chunk_id": cid,
                "fts_rank": fts_rank,
                "bm25": bm25,
                "vector_rank": vector_rank,
                "cosine": cosine,
                "fts_rrf_contribution": 1 / (60 + fts_rank) if fts_rank and fts_rank <= 60 else 0,
                "vector_rrf_contribution": 1 / (60 + vector_rank)
                if vector_rank and vector_rank <= 60
                else 0,
                "rrf_score": fused.get(cid, 0),
                "rrf_rank": fused_positions.get(cid),
                "in_legacy_top8": cid in fused_positions and fused_positions[cid] <= 8,
            }

        scope = ChatService(database, search=search)._question_scope(question, None, None)
        improved = search.search(question, document_ids=scope, limit=20, evidence=True)
        return {
            "query": question,
            "legacy_fts_query": fts_query,
            "target": details(chunk_id),
            "legacy_top8": [details(cid) for cid, rank in fused_positions.items() if rank <= 8],
            "evidence_top20_ids": [s["id"] for s in improved],
            "evidence_document_ids": scope,
        }
    finally:
        engine.dispose()
