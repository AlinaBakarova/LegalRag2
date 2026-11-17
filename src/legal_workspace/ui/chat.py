import json

from nicegui import run, ui

from legal_workspace.documents.service import DocumentService
from legal_workspace.llm.ollama import LLMError
from legal_workspace.rag.service import ChatService
from legal_workspace.ui.documents import log_failure

LOCATORS = {
    "page": "Страница",
    "line": "Строка",
    "paragraph": "Абзац",
    "table": "Таблица",
    "row": "Строка таблицы",
    "cell": "Ячейка",
}


def chat_panel(database):
    chat = ChatService(database)
    documents = DocumentService(database)
    current = None
    snapshot = None
    active_dialog = None
    active_citation = None
    ui.label("Чат по вашим документам").classes("text-lg font-semibold")
    ui.label(
        "Локальная модель отвечает по доступным источникам. Проверяйте выводы по цитатам."
    ).classes("text-sm muted")

    async def select_session(event):
        nonlocal current, snapshot
        current = event.value
        snapshot = None
        await refresh()

    async def new_session():
        nonlocal current, snapshot
        current = await run.io_bound(chat.create_session)
        snapshot = None
        await refresh()

    with ui.row().classes("w-full items-center gap-3"):
        sessions = ui.select({}, label="Диалог", on_change=select_session).classes("flex-1 min-w-0")
        ui.button("Новый диалог", icon="add", on_click=new_session).props("flat no-caps")

    async def delete_session():
        if not current:
            return
        session_id = current
        with ui.dialog() as dialog, ui.card():
            ui.label("Удалить диалог и ответы?")

            async def confirm():
                nonlocal current, snapshot
                await run.io_bound(chat.delete_session, session_id)
                current, snapshot = None, None
                dialog.close()
                await refresh()

            ui.button("Удалить", on_click=confirm, color="negative")
            ui.button("Отмена", on_click=dialog.close)
        dialog.open()

    with ui.button(icon="delete_outline", on_click=delete_session).props(
        'flat round dense aria-label="Удалить диалог"'
    ):
        ui.tooltip("Удалить диалог")
    scope = (
        ui.select({}, multiple=True, label="Документы для вопроса (пусто — вся библиотека)")
        .props("clearable use-chips")
        .classes("w-full")
    )
    with ui.scroll_area().classes("w-full h-[50vh]"):
        transcript = ui.column().classes("w-full gap-4")

    async def show_citation(citation_id):
        nonlocal active_dialog, active_citation
        try:
            citation = await run.io_bound(chat.citation, citation_id)
        except LLMError:
            ui.notify("Источник больше недоступен", type="warning")
            return
        with ui.dialog() as dialog, ui.card().classes("w-full max-w-3xl"):
            ui.label(citation["title"]).classes("text-lg break-words w-full")
            locator = " · ".join(
                f"{LOCATORS.get(k, k)} {v}" for k, v in citation["locator"].items() if k != "type"
            )
            ui.label(locator).classes("text-sm muted")
            if citation["heading"]:
                ui.label(citation["heading"]).classes("text-sm muted")
            ui.label(citation["quote"]).classes("whitespace-pre-wrap break-words w-full")
            page = citation["locator"].get("page")
            suffix = f"#page={page}" if page else ""
            ui.link(
                "Открыть оригинал",
                f"/api/documents/{citation['document_id']}/original?download=false{suffix}",
                new_tab=True,
            )
            ui.button("Закрыть", on_click=dialog.close)
        active_dialog, active_citation = dialog, citation_id
        dialog.open()

    async def retry(reply_id):
        try:
            await run.io_bound(chat.retry, reply_id)
        except LLMError:
            ui.notify("Не удалось повторить запрос", type="negative")
        await refresh()

    async def cancel(reply_id):
        await run.io_bound(chat.cancel, reply_id)
        await refresh()

    async def refresh():
        nonlocal current, snapshot, active_dialog, active_citation
        try:
            history = await run.io_bound(chat.sessions)
            options = {s["id"]: s["title"] for s in history}
            if current not in options:
                current = history[0]["id"] if history else None
            if sessions.options != options or sessions.value != current:
                sessions.set_options(options, value=current)
            available = await run.io_bound(documents.list)
            document_options = {d["id"]: d["source"].get("title", d["filename"]) for d in available}
            if scope.options != document_options:
                scope.set_options(
                    document_options,
                    value=[key for key in (scope.value or []) if key in document_options],
                )
            messages = await run.io_bound(chat.messages, current) if current else []
            if active_dialog is not None:
                try:
                    await run.io_bound(chat.citation, active_citation)
                except LLMError:
                    active_dialog.close()
                    active_dialog, active_citation = None, None
            pending = any(m["status"] in {"queued", "running"} for m in messages)
            send_button.set_enabled(not pending)
            latest = json.dumps(messages, sort_keys=True, ensure_ascii=False)
            if latest == snapshot:
                return
            snapshot = latest
            transcript.clear()
            with transcript:
                if not messages:
                    ui.label("Задайте вопрос по документам вашей библиотеки.").classes(
                        "muted text-sm"
                    )
                for message in messages:
                    with ui.card().classes("w-full shadow-none border gap-2"):
                        ui.label("Вы" if message["role"] == "user" else "Ответ").classes(
                            "font-medium text-sm"
                        )
                        status = message["status"]
                        if message["role"] == "user":
                            ui.label(message["content"]).classes(
                                "whitespace-pre-wrap break-words w-full"
                            )
                        elif status == "completed":
                            for ordinal, statement in enumerate(json.loads(message["content"])):
                                ui.label(statement).classes(
                                    "whitespace-pre-wrap break-words w-full"
                                )
                                for citation in message["citations"]:
                                    if citation["claim_ordinal"] == ordinal:
                                        ui.button(
                                            "Источник: " + citation["title"],
                                            on_click=lambda _e, cid=citation["id"]: show_citation(
                                                cid
                                            ),
                                        ).props("flat no-caps").classes("w-full text-left")
                            if message["usage"].get("partial_sources"):
                                ui.label(
                                    "Часть источников не удалось прочитать полностью."
                                ).classes("text-xs text-amber-900")
                            if message["usage"].get("partial_corpus"):
                                ui.label("Некоторые документы ещё недоступны для поиска.").classes(
                                    "text-xs text-amber-900"
                                )
                            ui.label("Применимость норм требует проверки.").classes("text-xs muted")
                        elif status == "insufficient":
                            ui.label(message["content"]).classes("text-sm muted")
                        elif status in {"queued", "running"}:
                            with ui.row().classes("items-center"):
                                ui.spinner(size="18px")
                                ui.label(
                                    "Вопрос ожидает обработки…"
                                    if status == "queued"
                                    else "Готовлю ответ по источникам…"
                                ).classes("text-sm muted")
                            ui.button(
                                "Отменить ответ", on_click=lambda _e, rid=message["id"]: cancel(rid)
                            ).props("flat no-caps")
                        elif status == "removed":
                            ui.label(
                                "Ответ удалён, поскольку использованный документ был удалён или "
                                "обработан заново."
                            ).classes("text-sm muted")
                        elif status == "cancelled":
                            ui.label("Ответ отменён").classes("text-sm muted")
                            ui.button(
                                "Повторить запрос",
                                on_click=lambda _e, rid=message["id"]: retry(rid),
                            ).props("flat no-caps")
                        else:
                            ui.label(message.get("error", "Не удалось сформировать ответ")).classes(
                                "text-sm text-red-800"
                            )
                            ui.button(
                                "Повторить запрос",
                                on_click=lambda _e, rid=message["id"]: retry(rid),
                            ).props("flat no-caps")
        except Exception as exc:
            log_failure("chat_refresh", exc)
            ui.notify("Не удалось обновить диалог", type="negative")

    async def send():
        nonlocal current, snapshot
        if not question.value or not question.value.strip():
            return
        send_button.disable()
        try:
            if current is None:
                current = await run.io_bound(chat.create_session)
            await run.io_bound(
                chat.send, current, question.value, list(scope.value) if scope.value else None
            )
            question.set_value("")
            snapshot = None
            await refresh()
        except LLMError:
            ui.notify(
                (
                    "Не удалось отправить вопрос. Дождитесь предыдущего ответа и "
                    "проверьте длину вопроса."
                ),
                type="negative",
            )
            send_button.enable()
        except Exception as exc:
            log_failure("chat_send", exc)
            ui.notify("Не удалось отправить вопрос", type="negative")
            send_button.enable()

    with ui.row().classes("w-full items-center gap-3"):
        question = (
            ui.textarea(label="Вопрос", placeholder="Задайте вопрос по документам")
            .props("outlined maxlength=1500 autogrow")
            .classes("flex-1 min-w-0")
        )
        send_button = ui.button(icon="arrow_upward", on_click=send).props(
            'round aria-label="Отправить"'
        )
    ui.label(
        "Генерация и поиск выполняются локально. Документы не отправляются платным API."
    ).classes("text-xs muted")
    ui.timer(2, refresh, immediate=True)
