import os
import shutil

import pytest
from playwright.sync_api import expect, sync_playwright


@pytest.mark.browser
def test_chat_composer_fits_viewport_and_has_no_retrieval_controls(runtime):
    executable = os.environ.get("LEGAL_WORKSPACE_CHROMIUM") or shutil.which("chromium")
    if not executable:
        pytest.fail("Chromium is required")
    runtime.start_web()
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=executable, headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(runtime.base_url)
        expect(page.get_by_text("Базовая библиотека", exact=True)).to_be_visible()
        expect(page.get_by_text("Мои документы", exact=True)).to_be_visible()
        expect(page.get_by_text("Чат по вашим документам", exact=True)).to_be_visible()
        expect(page.get_by_label("Вопрос", exact=True)).to_be_enabled()
        for label in ("Поисковый запрос", "Способ поиска", "Область поиска"):
            expect(page.get_by_label(label, exact=True)).to_have_count(0)
        expect(page.get_by_role("button", name="Найти", exact=True)).to_have_count(0)
        expect(page.get_by_text("Поиск по документам", exact=True)).to_have_count(0)
        for width, height in ((1920, 1080), (1366, 768), (1024, 768), (390, 844)):
            page.set_viewport_size({"width": width, "height": height})
            expect(page.get_by_label("Вопрос", exact=True)).to_be_in_viewport()
            expect(page.get_by_role("button", name="Отправить", exact=True)).to_be_in_viewport()
            composer = page.locator(".chat-composer").bounding_box()
            assert composer["y"] + composer["height"] <= height
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
            assert page.evaluate("document.documentElement.scrollHeight <= window.innerHeight")
            expect(
                page.get_by_text("Что вы хотите узнать по документам?", exact=True)
            ).to_be_visible()
        page.get_by_role("button", name="Документы", exact=True).click()
        expect(page.get_by_text("Мои документы", exact=True)).to_be_visible()
        page.get_by_role("button", name="Закрыть документы", exact=True).click()
        expect(page.get_by_label("Вопрос", exact=True)).to_be_in_viewport()
        page.get_by_role("button", name="Выбрать документы", exact=True).click()
        expect(page.get_by_label("Документы для вопроса", exact=True)).to_be_visible()
        page.get_by_role("button", name="Готово", exact=True).click()
        expect(page.get_by_label("Документы для вопроса", exact=True)).not_to_be_visible()
        assert not errors
        browser.close()
