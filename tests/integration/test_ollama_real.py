"""Real generation, real SQLite/search pipeline; fixed test embeddings isolate Qwen.
Use --ollama --real-model to also run the separate actual embedding model test.
No paid APIs or automatic downloads are used by this test.
"""

import httpx
import pytest
from search_support import FixedEmbedder

from legal_workspace.config import Settings
from legal_workspace.documents.service import DocumentService
from legal_workspace.jobs.queue import JobQueue
from legal_workspace.jobs.worker import Worker
from legal_workspace.persistence.database import Database
from legal_workspace.rag.service import ChatService
from legal_workspace.retrieval.service import SearchService


@pytest.mark.ollama
@pytest.mark.parametrize("definition", [False, True], ids=["contract", "definition-with-headings"])
def test_real_qwen_rag_answer_with_exact_citation(tmp_path, definition):
    configured = Settings()
    try:
        response = httpx.get(configured.ollama_url + "/api/tags", timeout=10, trust_env=False)
        response.raise_for_status()
    except httpx.HTTPError:
        pytest.fail("Start local Ollama before running --ollama")
    models = [m["name"] for m in response.json()["models"]]
    if configured.llm_model not in models:
        pytest.fail("Pull the configured local Qwen model before running --ollama")
    assert configured.llm_model.startswith("qwen3:"), "This integration test targets Qwen3"
    settings = configured.model_copy(update={"data_dir": tmp_path, "lease_seconds": 900})
    database = Database(settings)
    database.initialize()
    try:
        documents = DocumentService(database)
        content = (
            "СИНТЕТИЧЕСКИЙ ТЕСТОВЫЙ ДОГОВОР. НЕ НОРМАТИВНЫЙ АКТ.\n"
            "Статья 1. Срок оплаты\nСрок оплаты составляет 17 дней."
        )
        question = "Каков срок оплаты по тестовому договору? Укажи срок и процитируй источник."
        expected = "17"
        if definition:
            content = (
                "СИНТЕТИЧЕСКИЙ ТЕСТ. НЕ НОРМАТИВНЫЙ АКТ.\n"
                + "\n".join(f"Глава {i}. Административные штрафы" for i in range(1, 90))
                + "\nСтатья 3.5. Административный штраф\n"
                + "ТЕСТОВОЕ определение: административный штраф является денежным взысканием."
            )
            question = "Что такое административный штраф согласно тестовому документу?"
            expected = "денежн"
        doc_id = documents.upload("synthetic-example.txt", content.encode())
        Worker(database, settings).run_once()
        queue = JobQueue(database)
        search = SearchService(database, FixedEmbedder())
        claim = queue.claim("real-index", 900)
        queue.finish(claim, search.process(claim))
        chat = ChatService(database, search=search)
        session = chat.create_session()
        reply = chat.send(
            session,
            question,
            [doc_id],
        )
        claim = queue.claim("real-qwen", 900)
        result = chat.process(claim)
        message = chat.messages(session)[-1]
        assert message["status"] == "completed", message.get("error", result)
        assert expected in message["content"].casefold()
        assert message["model"] == settings.llm_model
        assert message["usage"]["calls"] <= settings.llm_max_calls
        assert message["citations"]
        for citation in message["citations"]:
            assert chat.citation(citation["id"])["quote"] == citation["quote"]
        assert reply == message["id"]
    finally:
        database.close()
