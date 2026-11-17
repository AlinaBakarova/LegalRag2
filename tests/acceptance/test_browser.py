import os
import shutil

import pytest
from conftest import wait_until
from playwright.sync_api import expect, sync_playwright


@pytest.mark.browser
def test_document_panels_and_durable_check_in_browser(runtime):
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
        expect(page.locator("input[type=file]")).to_have_count(1)
        expect(page.get_by_text("Рабочий процесс отключён", exact=True)).to_be_visible()
        page.get_by_role("button", name="Проверить рабочий процесс").click()
        expect(page.get_by_text("Ожидает запуска", exact=True)).to_be_visible()
        runtime.start("worker")
        expect(page.get_by_text("Проверка пройдена", exact=True)).to_be_visible(timeout=15000)
        page.locator("input[type=file]").set_input_files(
            {"name": "browser.txt", "mimeType": "text/plain", "buffer": "Проверка текста".encode()}
        )
        expect(page.get_by_text("Текст извлечён", exact=True)).to_be_visible(timeout=15000)
        page.get_by_role("button", name="browser.txt", exact=True).click()
        expect(page.get_by_text("Проверка текста", exact=True)).to_be_visible()
        page.get_by_role("button", name="Закрыть", exact=True).click()
        page.get_by_role("button", name="Удалить документ", exact=True).click()
        page.get_by_role("button", name="Удалить", exact=True).click()
        expect(page.get_by_role("button", name="browser.txt", exact=True)).to_have_count(0)
        page.reload()
        expect(page.get_by_text("Проверка пройдена", exact=True)).to_be_visible()
        page.set_viewport_size({"width": 390, "height": 844})
        wait_until(
            lambda: page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        )
        assert not errors
        browser.close()
