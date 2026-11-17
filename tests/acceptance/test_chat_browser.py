import os
import shutil
from pathlib import Path

import pytest
from playwright.sync_api import expect, sync_playwright
from search_support import FixedEmbedder

from legal_workspace.documents.service import DocumentService
from legal_workspace.jobs.queue import JobQueue
from legal_workspace.jobs.worker import Worker
from legal_workspace.persistence.database import Database
from legal_workspace.retrieval.service import SearchService


@pytest.mark.browser
def test_chat_sources_restart_and_deletion(runtime):
    executable = os.environ.get("LEGAL_WORKSPACE_CHROMIUM") or shutil.which("chromium")
    if not executable:
        pytest.fail("Chromium required")
    db = Database(runtime.settings)
    db.initialize()
    doc_id = DocumentService(db).upload(
        "Договор.txt",
        "СИНТЕТИЧЕСКИЙ ТЕСТ.\nСтатья 1. Оплата\nСрок оплаты составляет 17 дней.".encode(),
    )
    Worker(db, runtime.settings).run_once()
    queue = JobQueue(db)
    claim = queue.claim("indexer", 120)
    queue.finish(claim, SearchService(db, FixedEmbedder()).process(claim))
    support = str(Path(__file__).parents[1])
    setup = f"""import sys
sys.path.insert(0, {support!r})
from search_support import FixedEmbedder
from rag_support import MockLLM
import legal_workspace.retrieval.service as retrieval
import legal_workspace.rag.service as rag
retrieval.get_embedder = lambda *_args: FixedEmbedder()
rag.OllamaGenerator = lambda *_args: MockLLM()
from legal_workspace.config import Settings
settings=Settings()
"""
    runtime.start_web(
        code=setup
        + """from legal_workspace.main import create_app
import uvicorn
uvicorn.run(create_app(settings),host=settings.host,port=settings.port)
"""
    )
    runtime.start(
        "worker",
        code=setup
        + """from legal_workspace.persistence.database import Database
from legal_workspace.jobs.worker import Worker
database=Database(settings)
database.initialize()
Worker(database,settings).run()
""",
    )
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path=executable, headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1100})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(runtime.base_url)
            page.get_by_label("Вопрос", exact=True).fill("Каков срок оплаты?")
            page.get_by_role("button", name="Отправить", exact=True).click()
            expect(page.get_by_text("Срок оплаты составляет 17 дней.", exact=True)).to_be_visible(
                timeout=15000
            )
            page.get_by_role("button", name="Источник: Договор.txt", exact=True).click()
            expect(page.get_by_role("link", name="Открыть оригинал", exact=True)).to_be_visible()
            page.get_by_role("button", name="Закрыть", exact=True).click()
            page.reload()
            expect(page.get_by_text("Срок оплаты составляет 17 дней.", exact=True)).to_be_visible()
            DocumentService(db).delete(doc_id)
            expect(page.get_by_text("Ответ удалён, поскольку", exact=False)).to_be_visible(
                timeout=10000
            )
            expect(page.get_by_text("Срок оплаты составляет 17 дней.", exact=True)).to_have_count(0)
            assert not errors
            browser.close()
    finally:
        db.close()
