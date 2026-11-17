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
def test_search_excerpt_source_and_deleted_result(runtime):
    executable = os.environ.get("LEGAL_WORKSPACE_CHROMIUM") or shutil.which("chromium")
    if not executable:
        pytest.fail("Chromium is required")
    database = Database(runtime.settings)
    database.initialize()
    documents = DocumentService(database)
    doc_id = documents.upload(
        "Тестовый договор.txt",
        (
            "СИНТЕТИЧЕСКИЙ ПРИМЕР.\nСтатья 5. Неустойка\nЗа просрочку начисляется тестовый штраф."
        ).encode(),
    )
    Worker(database, runtime.settings).run_once()
    claim = JobQueue(database).claim("test-indexer", 120)
    search = SearchService(database, FixedEmbedder())
    JobQueue(database).finish(claim, search.process(claim))
    # Browser infrastructure test uses declared fixed test vectors. Actual ONNX
    # quality and download are checked separately by --real-model and the evaluator.
    support = Path(__file__).parents[1]
    code = f"""
import sys
sys.path.insert(0, {str(support)!r})
from search_support import FixedEmbedder
import legal_workspace.retrieval.service as retrieval
retrieval.get_embedder = lambda *_args: FixedEmbedder()
from legal_workspace.main import create_app
from legal_workspace.config import Settings
import uvicorn
settings = Settings()
uvicorn.run(create_app(settings), host=settings.host, port=settings.port)
"""
    runtime.start_web(code=code)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path=executable, headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(runtime.base_url)
            page.get_by_label("Поисковый запрос", exact=True).fill("неустойки за просрочку")
            page.get_by_role("button", name="Найти", exact=True).click()
            expect(page.get_by_text("Найдено фрагментов:", exact=False)).to_be_visible()
            page.get_by_role("button", name="Источник", exact=True).first.click()
            expect(page.get_by_role("link", name="Открыть оригинал", exact=True)).to_be_visible()
            expect(page.get_by_text("Строка", exact=False).last).to_be_visible()
            page.get_by_role("button", name="Закрыть", exact=True).click()
            documents.delete(doc_id)
            expect(page.get_by_role("button", name="Источник", exact=True)).to_have_count(
                0, timeout=10000
            )
            expect(
                page.get_by_text("Состав документов изменился. Повторите поиск.", exact=True)
            ).to_be_visible()
            page.get_by_role("button", name="Найти", exact=True).click()
            expect(page.get_by_text("Совпадений не найдено.", exact=False)).to_be_visible()
            assert not errors
            browser.close()
    finally:
        database.close()
