from nicegui import run, ui

from legal_workspace.retrieval.embeddings import SearchError
from legal_workspace.retrieval.service import SearchService

LOCATOR_NAMES = {
    "page": "Страница",
    "line": "Строка",
    "paragraph": "Абзац",
    "table": "Таблица",
    "row": "Строка таблицы",
    "cell": "Ячейка",
}


def source_label(item):
    return " · ".join(
        f"{LOCATOR_NAMES.get(key, key)} {value}"
        for key, value in item["locator"].items()
        if key != "type"
    )


def search_panel(database):
    service = SearchService(database)
    ui.label("Поиск по документам").classes("text-lg font-semibold")
    ui.label("Найдите нужное положение и откройте его источник.").classes("text-sm muted")
    from legal_workspace.documents.service import DocumentService

    documents = DocumentService(database)
    # Options refresh on demand, so uploaded/deleted files are reflected without a page reload.
    scope = ui.select(
        {"all": "Все документы", "base": "Базовая библиотека", "user": "Мои документы"},
        value="all",
        label="Область поиска",
    ).classes("w-full")
    mode = ui.select(
        {"hybrid": "По смыслу и словам", "lexical": "По словам"},
        value="hybrid",
        label="Способ поиска",
    ).classes("w-full")
    selected = (
        ui.select({}, multiple=True, label="Выбранные документы (пусто — все)")
        .props("clearable use-chips")
        .classes("w-full")
    )

    last_corpus = None

    async def refresh_options():
        nonlocal last_corpus
        current = await run.io_bound(documents.list)
        options = {d["id"]: d["source"].get("title", d["filename"]) for d in current}
        snapshot = [
            (d["id"], d["text_revision"], d["active_index_id"], d["status"]) for d in current
        ]
        if last_corpus is not None and snapshot != last_corpus:
            results.clear()
            feedback.set_text("Состав документов изменился. Повторите поиск.")
        last_corpus = snapshot
        status = await run.io_bound(service.status)
        if not status["model_ready"]:
            readiness.set_text("Поиск пока недоступен: требуется подготовка поисковой модели.")
        elif status["failed"]:
            readiness.set_text("Не удалось подготовить часть документов к поиску.")
        elif status["indexed"] < status["available"]:
            readiness.set_text(
                f"Документы готовятся к поиску: {status['indexed']} из {status['available']}."
            )
        else:
            readiness.set_text("")
        retry_button.set_visibility(status["model_ready"] and bool(status["failed"]))
        if selected.options != options:
            selected.set_options(
                options, value=[key for key in (selected.value or []) if key in options]
            )

    async def retry_index():
        try:
            await run.io_bound(service.schedule_missing)
            await refresh_options()
        except Exception as exc:
            from legal_workspace.ui.documents import log_failure

            log_failure("retry_index", exc)
            ui.notify("Не удалось подготовить поиск. Попробуйте ещё раз.", type="negative")

    readiness = ui.label().classes("text-sm muted")
    retry_button = ui.button("Повторить подготовку поиска", on_click=retry_index).props(
        "flat no-caps"
    )
    retry_button.set_visibility(False)
    ui.timer(3, refresh_options, immediate=True)
    query = (
        ui.input(label="Поисковый запрос", placeholder="Например: сроки исполнения обязательств")
        .props("outlined maxlength=1500")
        .classes("w-full")
    )
    with ui.row().classes("items-center"):
        button = ui.button("Найти", icon="search").props("no-caps")
        spinner = ui.spinner(size="20px")
        spinner.set_visibility(False)
    feedback = ui.label().classes("text-sm muted")
    results = ui.column().classes("w-full gap-4")

    async def show_source(chunk_id):
        try:
            item = await run.io_bound(service.source, chunk_id)
        except SearchError:
            ui.notify("Источник больше недоступен. Повторите поиск.", type="warning")
            return
        with ui.dialog() as dialog, ui.card().classes("w-full max-w-3xl"):
            ui.label(item["title"]).classes("text-lg break-words w-full")
            ui.label(source_label(item)).classes("text-sm muted")
            if item["heading"]:
                ui.label(item["heading"]).classes("text-sm font-medium")
            # Only the matched source excerpt is shown here. Opening a document in
            # the file panel continues to show its original, not an extracted full text.
            ui.label(item["text"]).classes("whitespace-pre-wrap break-words w-full")
            if item["warnings"]:
                ui.label("Часть документа не удалось прочитать").classes("text-amber-900 text-sm")
            page = item["locator"].get("page")
            suffix = f"#page={page}" if page else ""
            ui.link(
                "Открыть оригинал",
                f"/api/documents/{item['document_id']}/original?download=false{suffix}",
                new_tab=True,
            )
            ui.button("Закрыть", on_click=dialog.close).props("no-caps")
        dialog.open()

    async def search():
        if not query.value or not query.value.strip():
            feedback.set_text("Введите поисковый запрос")
            return
        button.disable()
        spinner.set_visibility(True)
        results.clear()
        feedback.set_text("")
        try:
            current_documents = await run.io_bound(documents.list)
            ids = (
                None
                if scope.value == "all"
                else [d["id"] for d in current_documents if d["kind"] == scope.value]
            )
            if selected.value:
                ids = [key for key in selected.value if ids is None or key in ids]
            found = await run.io_bound(
                service.search, query.value, mode=mode.value, document_ids=ids
            )
            with results:
                for item in found:
                    with ui.card().classes("w-full shadow-none border gap-2"):
                        ui.label(item["title"]).classes("font-medium break-words w-full")
                        ui.label(source_label(item)).classes("text-xs muted")
                        if item["heading"]:
                            ui.label(item["heading"]).classes("text-sm muted")
                        ui.label(
                            item["text"][:350] + ("…" if len(item["text"]) > 350 else "")
                        ).classes("text-sm whitespace-pre-wrap break-words w-full")
                        ui.button(
                            "Источник",
                            on_click=lambda _event, chunk_id=item["id"]: show_source(chunk_id),
                        ).props("flat no-caps")
            feedback.set_text(
                f"Найдено фрагментов: {len(found)}"
                if found
                else "Совпадений не найдено. Новые документы могут ещё обрабатываться."
            )
        except SearchError as exc:
            from legal_workspace.ui.documents import log_failure

            log_failure("search", exc)
            feedback.set_text(
                "Поиск пока недоступен: требуется подготовка поисковой модели."
                if str(exc) == "model_not_ready"
                else "Не удалось выполнить поиск. Попробуйте изменить запрос."
            )
        except Exception as exc:
            from legal_workspace.ui.documents import log_failure

            log_failure("search", exc)
            feedback.set_text("Не удалось выполнить поиск. Попробуйте ещё раз.")
        finally:
            spinner.set_visibility(False)
            button.enable()

    button.on_click(search)
    query.on("keydown.enter", search)
    ui.label(
        "Результаты поиска — фрагменты источников. Юридическая оценка пока не выполняется."
    ).classes("text-xs muted")
