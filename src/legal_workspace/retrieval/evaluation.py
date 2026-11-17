"""Runs against a temporary corpus. Never changes the user's documents or indices."""

import json
import math
import tempfile
import time
from pathlib import Path

from legal_workspace.config import Settings
from legal_workspace.documents.service import DocumentService
from legal_workspace.jobs.queue import JobQueue
from legal_workspace.jobs.worker import Worker
from legal_workspace.persistence.database import Database
from legal_workspace.retrieval.embeddings import get_embedder
from legal_workspace.retrieval.service import SearchService


def metrics(ranking, relevant, k=5):
    ranking = list(dict.fromkeys(ranking))[:k]
    relevant = set(relevant)
    hits = [int(item in relevant) for item in ranking]
    first = next((index for index, hit in enumerate(hits, 1) if hit), None)
    dcg = sum(hit / math.log2(index + 1) for index, hit in enumerate(hits, 1))
    ideal = sum(1 / math.log2(index + 1) for index in range(1, min(k, len(relevant)) + 1))
    return {
        "recall@5": sum(hits) / len(relevant) if relevant else 0,
        "mrr@5": 1 / first if first else 0,
        "ndcg@5": dcg / ideal if ideal else 0,
    }


def evaluate(data_dir: Path, embedder=None):
    model = embedder if embedder is not None else get_embedder(data_dir)
    dataset = json.loads((Path(__file__).parent / "fixtures/evaluation.json").read_text())
    with tempfile.TemporaryDirectory(prefix="legal-retrieval-eval-") as temporary:
        settings = Settings(data_dir=Path(temporary), _env_file=None)
        database = Database(settings)
        database.initialize()
        documents = DocumentService(database)
        search = SearchService(database, model)
        keys = {}
        try:
            for source in dataset["documents"]:
                document_id = documents.upload(source["key"] + ".txt", source["text"].encode())
                keys[document_id] = source["key"]
                Worker(database, settings).run_once()
            queue = JobQueue(database)
            while claim := queue.claim("evaluation", 300):
                if claim.kind != "search.index":
                    raise RuntimeError("Unexpected evaluation job")
                queue.finish(claim, search.process(claim))
            report = {
                "dataset": "synthetic-fixture-v1",
                "metrics_unit": "document",
                "model": model.identity,
                "notice": dataset["notice"],
                "modes": {},
            }
            for mode in ("lexical", "semantic", "hybrid"):
                rows = []
                started = time.monotonic()
                for query in dataset["queries"]:
                    found = search.search(query["text"], mode=mode, limit=20)
                    scores = metrics(
                        [keys[item["document_id"]] for item in found], query["relevant"]
                    )
                    rows.append({"query": query["text"], **scores})
                report["modes"][mode] = {
                    "queries": len(rows),
                    "means": {
                        key: sum(row[key] for row in rows) / len(rows)
                        for key in ("recall@5", "mrr@5", "ndcg@5")
                    },
                    "seconds": time.monotonic() - started,
                    "results": rows,
                }
            return json.dumps(report, ensure_ascii=False, indent=2)
        finally:
            database.close()
