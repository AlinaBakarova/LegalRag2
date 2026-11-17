import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from search_support import FixedEmbedder
from sqlalchemy import func, select, text, update

from legal_workspace.documents.service import DocumentService
from legal_workspace.jobs.queue import JobQueue
from legal_workspace.jobs.worker import Worker
from legal_workspace.persistence.database import Database
from legal_workspace.persistence.models import Chunk, Document, IndexGeneration, Job, SourceSpan
from legal_workspace.retrieval.chunking import chunk_span, normalize
from legal_workspace.retrieval.embeddings import SearchError
from legal_workspace.retrieval.service import SearchService


@pytest.fixture
def search(database):
    return SearchService(database, FixedEmbedder())


def add_indexed(database, search, settings, value, name="test.txt"):
    documents = DocumentService(database)
    document_id = documents.upload(name, value.encode())
    Worker(database, settings).run_once()
    claim = JobQueue(database).claim("test-indexer", 120)
    assert claim.kind == "search.index"
    result = search.process(claim)
    assert result["indexed"]
    assert JobQueue(database).finish(claim, result)
    return document_id


def test_chunk_offsets_and_headings():
    value = (
        "Статья 1. Тестовые условия\n"
        + "Предложение об оплате. " * 100
        + "\nСтатья 2. Другой раздел\nКонфиденциальность."
    )
    pieces = list(chunk_span({"id": 5, "text": value}, "", FixedEmbedder()))
    assert len(pieces) > 2
    assert all(value[p.start : p.end] == p.text for p in pieces)
    assert pieces[-1].heading.startswith("Статья 2.")
    assert all(len(p.text) <= 900 for p in pieces)
    covered = set().union(*(set(range(p.start, p.end)) for p in pieces))
    assert len(covered) == len(value)


def test_russian_morphology_and_numeric_references():
    assert normalize("неустойки") == normalize("неустойка")
    assert normalize("СТ. 330") == normalize("статья 330")
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert len(set(pool.map(normalize, ["Платежи и обязательства"] * 40))) == 1


def test_lexical_semantic_fusion_and_source_offsets(database, search, settings):
    penalty = add_indexed(
        database,
        search,
        settings,
        "ТЕСТОВЫЙ ДОКУМЕНТ, НЕ ЗАКОН.\nСтатья 330. Неустойка\n"
        "За просрочку тестового платежа назначается тестовый штраф.",
        "penalty.txt",
    )
    add_indexed(
        database,
        search,
        settings,
        "ТЕСТОВЫЙ ДОКУМЕНТ.\nСтатья 1. Срок исполнения\nСрок составляет десять дней.",
        "deadline.txt",
    )
    for mode in ("lexical", "semantic", "hybrid"):
        result = search.search("неустойки за просрочку", mode=mode)
        assert result[0]["document_id"] == penalty
        source = search.source(result[0]["id"])
        spans = DocumentService(database).spans(penalty)
        assert any(
            span["text"][source["start"] : source["end"]] == source["text"] for span in spans
        )
    assert search.search("неустойки", mode="lexical", document_ids=[]) == []
    assert all(
        item["document_id"] == penalty for item in search.search("срок", document_ids=[penalty])
    )
    assert search.search("абракадабрасовпаденийнебудет", mode="lexical") == []


def test_add_delete_cleans_fts_embeddings_and_job(database, search, settings):
    documents = DocumentService(database)
    doc_id = add_indexed(database, search, settings, "Конфиденциальная тайна.", "secret.txt")
    chunk_id = search.search("тайна", mode="lexical")[0]["id"]
    documents.delete(doc_id)
    assert search.search("тайна") == []
    with pytest.raises(SearchError, match="source_not_found"):
        search.source(chunk_id)
    with database.engine.connect() as connection:
        for model in (Document, SourceSpan, Chunk, IndexGeneration):
            assert connection.scalar(select(func.count()).select_from(model)) == 0
        assert (
            connection.execute(
                text("SELECT count(*) FROM chunk_fts WHERE chunk_fts MATCH 'тайн'")
            ).scalar_one()
            == 0
        )
        connection.exec_driver_sql(
            "INSERT INTO chunk_fts(chunk_fts, rank) VALUES('integrity-check', 1)"
        )


def test_persistence_and_no_reindex_when_config_unchanged(database, search, settings):
    add_indexed(database, search, settings, "Срок оплаты десять дней.")
    assert search.schedule_missing() == 0
    database.close()
    reopened = Database(settings)
    try:
        reopened.initialize()
        again = SearchService(reopened, FixedEmbedder())
        assert again.search("оплата")
        assert again.schedule_missing() == 0
    finally:
        reopened.close()


def test_failed_rebuild_preserves_active_generation(database, search, settings, monkeypatch):
    add_indexed(database, search, settings, "Неустойка за просрочку.")
    before = search.search("неустойка")
    search.schedule_missing(force=True)
    claim = JobQueue(database).claim("rebuild", 120)

    def broken(_texts):
        raise RuntimeError("TEST failure")

    with monkeypatch.context() as patch:
        patch.setattr(search.embedder, "embed", broken)
        with pytest.raises(RuntimeError):
            search.process(claim)
    assert search.search("неустойка") == before
    with database.engine.connect() as connection:
        assert connection.scalar(select(func.count()).select_from(IndexGeneration)) == 1


def test_partial_generation_not_visible_and_deleted_during_build(
    database, search, settings, monkeypatch
):
    documents = DocumentService(database)
    document_id = documents.upload("long.txt", ("Тайна и конфиденциальность. " * 1500).encode())
    Worker(database, settings).run_once()
    claim = JobQueue(database).claim("builder", 120)
    embed = search.embedder.embed
    calls = 0

    def delete_while_embedding(texts):
        nonlocal calls
        calls += 1
        with pytest.raises(SearchError, match="index_not_ready"):
            search.search("тайна", mode="lexical")
        if calls == 2:
            documents.delete(document_id)
        return embed(texts)

    monkeypatch.setattr(search.embedder, "embed", delete_while_embedding)
    with pytest.raises(SearchError, match="stale_index_job"):
        search.process(claim)
    assert calls == 2
    assert documents.list() == []
    with database.engine.connect() as connection:
        assert connection.scalar(select(func.count()).select_from(Chunk)) == 0


def test_model_version_and_dimension_are_isolated(database, search, settings):
    add_indexed(database, search, settings, "Срок оплаты.")
    other = FixedEmbedder()
    other.identity = "different-model"
    different = SearchService(database, other)
    with pytest.raises(SearchError, match="index_not_ready"):
        different.search("оплата")
    assert different.schedule_missing() == 1
    old = search.search("оплата")[0]["id"]
    claim = JobQueue(database).claim("new-model", 120)
    different.process(claim)
    assert different.search("оплата")
    with pytest.raises(SearchError, match="index_not_ready"):
        search.search("оплата")
    with pytest.raises(SearchError):
        search.source(old)


def test_reprocessing_invalidates_old_sources(database, search, settings):
    doc_id = add_indexed(database, search, settings, "Десять дней — срок оплаты.")
    old_id = search.search("оплата")[0]["id"]
    DocumentService(database).reprocess(doc_id)
    with pytest.raises(SearchError, match="index_not_ready"):
        search.search("оплата")
    with pytest.raises(SearchError):
        search.source(old_id)


def test_lost_lease_and_abandoned_generation_recovery(database, search, settings):
    doc_id = DocumentService(database).upload("test.txt", b"content")
    Worker(database, settings).run_once()
    claim = JobQueue(database).claim("old-worker", 120)
    with database.transaction() as connection:
        revision = connection.scalar(select(Document.text_revision).where(Document.id == doc_id))
        connection.execute(
            IndexGeneration.__table__.insert().values(
                id="a" * 32,
                document_id=doc_id,
                text_revision=revision,
                fingerprint=search.fingerprint,
                dimension=5,
                state="building",
                job_id=claim.id,
                lease_token=claim.token,
                created_at=time.time(),
            )
        )
    search.recover()
    with database.engine.connect() as connection:
        assert connection.scalar(select(func.count()).select_from(IndexGeneration)) == 1
    with database.transaction() as connection:
        connection.execute(update(Job).where(Job.id == claim.id).values(lease_expires_at=0))
    search.recover()
    with database.engine.connect() as connection:
        assert connection.scalar(select(func.count()).select_from(IndexGeneration)) == 0
    with pytest.raises(SearchError, match="stale_index_job"):
        search.process(claim)


def test_search_api(database, settings, search):
    from fastapi.testclient import TestClient

    from legal_workspace.api import create_api

    doc_id = add_indexed(database, search, settings, "Неустойка за просрочку.")
    with TestClient(create_api(settings), base_url="http://127.0.0.1") as client:
        client.app.state.search_service._embedder = search.embedder
        response = client.post("/api/search", json={"query": "неустойка"})
        assert response.status_code == 200
        chunk_id = response.json()[0]["id"]
        assert client.get(f"/api/search/chunks/{chunk_id}").status_code == 200
        assert client.post("/api/search", json={"query": ""}).status_code == 422
        assert client.post("/api/search", json={"query": "a" * 1501}).status_code == 422
        assert client.delete(f"/api/documents/{doc_id}").status_code == 204
        assert client.get(f"/api/search/chunks/{chunk_id}").status_code == 404
        assert client.post("/api/search", json={"query": "неустойка"}).json() == []


def test_missing_model_returns_explicit_error(settings):
    from fastapi.testclient import TestClient

    from legal_workspace.api import create_api

    with TestClient(create_api(settings), base_url="http://127.0.0.1") as client:
        response = client.post("/api/search", json={"query": "оплата"})
        assert response.status_code == 503
        assert response.json()["detail"] == "model_not_ready"


def test_evaluation_math_and_runner(settings):
    import json

    from legal_workspace.retrieval.evaluation import evaluate, metrics

    assert metrics(["irrelevant", "yes", "yes"], ["yes"])["mrr@5"] == 0.5
    assert metrics([], ["yes"])["recall@5"] == 0
    assert metrics(["yes"], ["yes"])["ndcg@5"] == 1
    report = json.loads(evaluate(settings.data_dir, FixedEmbedder()))
    assert set(report["modes"]) == {"lexical", "semantic", "hybrid"}
    assert all(mode["queries"] == 5 for mode in report["modes"].values())
    assert "TEST-ONLY" in report["model"]


def test_mismatched_dimensions_and_corrupt_vector(database, search, settings):
    add_indexed(database, search, settings, "Неустойка за просрочку.")
    wider = FixedEmbedder()
    wider.dimension = 6
    # Even if an adapter accidentally reuses its identity, dimension is part of the fingerprint.
    with pytest.raises(SearchError, match="index_not_ready"):
        SearchService(database, wider).search("неустойка", mode="lexical")
    with database.transaction() as connection:
        connection.execute(update(Chunk).values(embedding=b"bad"))
    with pytest.raises(SearchError, match="incompatible_index"):
        search.search("неустойка")


def test_upgrade_stage_two_documents_get_revision_and_can_be_indexed(settings):
    from pathlib import Path

    from alembic import command
    from alembic.config import Config

    from legal_workspace.persistence import database as database_module

    settings.data_dir.mkdir(parents=True)
    database = Database(settings)
    config = Config()
    config.set_main_option(
        "script_location", str(Path(database_module.__file__).parent / "migrations")
    )
    doc_id = "f" * 32
    try:
        with database.engine.begin() as connection:
            config.attributes["connection"] = connection
            command.upgrade(config, "0002_documents")
            connection.execute(
                text("""INSERT INTO documents
                (id,kind,filename,sha256,size,status,created_at,warnings,source)
                VALUES (:id,'user','legacy.txt',:sha,10,'ready',0,'[]','{}')"""),
                {"id": doc_id, "sha": "a" * 64},
            )
            connection.execute(
                text("""INSERT INTO source_spans
                (document_id,ordinal,text,locator,start,end)
                VALUES (:id,0,'Неустойка',:locator,0,9)"""),
                {"id": doc_id, "locator": '{"type":"line","line":1}'},
            )
        database.initialize()
        documents = DocumentService(database)
        assert documents.get(doc_id)["text_revision"]
        search = SearchService(database, FixedEmbedder())
        assert search.schedule_missing() == 1
        claim = JobQueue(database).claim("migration-test", 120)
        search.process(claim)
        assert search.search("неустойка")[0]["document_id"] == doc_id
    finally:
        database.close()


def test_worker_indexes_after_extraction(database, settings, monkeypatch):
    import legal_workspace.retrieval.service as retrieval

    monkeypatch.setattr(retrieval, "get_embedder", lambda *args: FixedEmbedder())
    doc_id = DocumentService(database).upload("new.txt", "Неустойка за просрочку".encode())
    worker = Worker(database, settings)
    assert worker.run_once()  # extraction creates an indexing job in the same transaction
    assert worker.run_once()  # actual worker dispatch for indexing
    assert not worker.run_once()
    assert SearchService(database).search("неустойки")[0]["document_id"] == doc_id


def test_pending_index_is_not_a_negative_search_result(database, settings):
    from fastapi.testclient import TestClient

    from legal_workspace.api import create_api

    documents = DocumentService(database)
    doc_id = documents.upload("pending.txt", "Неустойка за просрочку".encode())
    Worker(database, settings).run_once()
    with TestClient(create_api(settings), base_url="http://127.0.0.1") as client:
        client.app.state.search_service._embedder = FixedEmbedder()
        status = client.get("/api/search/status").json()
        assert status["indexed"] == 0 and status["available"] == 1
        response = client.post("/api/search", json={"query": "неустойка"})
        assert response.status_code == 503
        assert response.json()["detail"] == "index_not_ready"
        # Empty or deleted scope still has ordinary empty results, not a readiness error.
        assert (
            client.post("/api/search", json={"query": "неустойка", "document_ids": []}).json() == []
        )
        client.delete(f"/api/documents/{doc_id}")
        assert client.post("/api/search", json={"query": "неустойка"}).json() == []
