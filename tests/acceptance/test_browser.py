import io
import os
import shutil

import pytest
from conftest import wait_until
from docx import Document
from playwright.sync_api import expect, sync_playwright


@pytest.mark.browser
def test_document_icons_states_and_original_only(runtime):
    executable = os.environ.get("LEGAL_WORKSPACE_CHROMIUM") or shutil.which("chromium")
    if not executable:
        pytest.fail("Set LEGAL_WORKSPACE_CHROMIUM to an installed Chromium executable")
    runtime.start_web()
    errors = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=executable, headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(runtime.base_url)
        expect(page.get_by_text("Базовая библиотека", exact=True)).to_be_visible()
        expect(page.get_by_text("Мои документы", exact=True)).to_be_visible()
        page.locator('[aria-label="Системные документы — недоступны для редактирования"]').hover()
        expect(
            page.get_by_text("Системные документы — недоступны для редактирования", exact=True)
        ).to_be_visible()
        expect(page.get_by_role("button", name="Проверить рабочий процесс")).to_have_count(0)
        page.locator("input[type=file]").set_input_files(
            {
                "name": "browser.txt",
                "mimeType": "text/plain",
                "buffer": "НЕ ПОКАЗЫВАТЬ ИЗВЛЕЧЁННЫЙ ТЕКСТ".encode(),
            }
        )
        expect(page.get_by_role("button", name="browser.txt", exact=True)).to_be_visible()
        expect(page.locator(".q-spinner")).to_be_visible()
        runtime.start("worker")
        wait_until(lambda: runtime.client.get("/api/documents").json()[0]["status"] == "ready")
        expect(page.locator(".q-spinner")).to_have_count(0, timeout=10000)
        expect(page.get_by_text("Текст извлечён", exact=True)).to_have_count(0)
        expect(page.get_by_text("Защищённый источник", exact=True)).to_have_count(0)
        page.get_by_role("button", name="browser.txt", exact=True).click()
        expect(page.get_by_text("НЕ ПОКАЗЫВАТЬ ИЗВЛЕЧЁННЫЙ ТЕКСТ", exact=True)).to_have_count(0)
        expect(page.get_by_role("link", name="Скачать оригинал")).to_be_visible()
        page.get_by_role("button", name="Закрыть", exact=True).click()
        trash = page.get_by_role("button", name="Удалить документ", exact=True)
        trash.hover()
        expect(page.get_by_text("Удалить документ", exact=True)).to_be_visible()
        trash.click()
        page.get_by_role("button", name="Удалить", exact=True).click()
        expect(page.get_by_role("button", name="browser.txt", exact=True)).to_have_count(0)
        page.locator("input[type=file]").set_input_files(
            {"name": "invalid.pdf", "mimeType": "application/pdf", "buffer": b"broken"}
        )
        expect(page.get_by_text("Не удалось обработать документ", exact=True)).to_be_visible(
            timeout=15000
        )
        expect(page.get_by_text("invalid_pdf", exact=True)).to_have_count(0)
        source = Document()
        source.add_paragraph("Основной текст")
        source.sections[0].header.paragraphs[0].text = "Текст колонтитула"
        data = io.BytesIO()
        source.save(data)
        page.locator("input[type=file]").set_input_files(
            {
                "name": "partial.docx",
                "mimeType": (
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                ),
                "buffer": data.getvalue(),
            }
        )
        expect(page.get_by_text("Часть документа не удалось прочитать", exact=True)).to_be_visible(
            timeout=15000
        )
        expect(page.get_by_text("docx_additional_parts_not_extracted", exact=True)).to_have_count(0)
        page.set_viewport_size({"width": 390, "height": 844})
        wait_until(
            lambda: page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        )
        assert not errors
        browser.close()
