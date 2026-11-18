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
    sending = False
    pending = False

    async def select_session(event):
        nonlocal current, snapshot
        current = event.value
        snapshot = None
        history_dialog.close()
        await refresh()

    async def new_session():
        nonlocal current, snapshot
        current = await run.io_bound(chat.create_session)
        snapshot = None
        await refresh()

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

    with ui.dialog() as history_dialog, ui.card().classes("w-80 max-w-full gap-4"):
        ui.label("Диалоги").classes("text-lg font-semibold")
        sessions = ui.select({}, label="Диалог", on_change=select_session).classes("w-full")
        with ui.row().classes("w-full items-center justify-between"):
            ui.button("Удалить диалог", icon="delete_outline", on_click=delete_session).props(
                "flat no-caps"
            )
            ui.button("Закрыть", on_click=history_dialog.close).props("flat no-caps")

    with ui.dialog() as scope_dialog, ui.card().classes("w-96 max-w-full gap-4"):
        ui.label("Источники для ответа").classes("text-lg font-semibold")
        ui.label("По умолчанию используются все документы библиотеки.").classes("text-sm muted")
        scope = (
            ui.select({}, multiple=True, label="Документы для вопроса")
            .props("clearable use-chips")
            .classes("w-full")
        )
        ui.button("Готово", on_click=scope_dialog.close).props("flat no-caps")

    with ui.row().classes("w-full items-center chat-toolbar"):
        ui.label("Чат по вашим документам").classes("font-semibold chat-toolbar-title")
        for icon, label, callback in (
            ("history", "История диалогов", history_dialog.open),
            ("tune", "Выбрать документы", scope_dialog.open),
            ("edit_square", "Новый диалог", new_session),
        ):
            with ui.button(icon=icon, on_click=callback).props(
                f'flat round dense aria-label="{label}"'
            ):
                ui.tooltip(label)

    with ui.element("div").classes("transcript-scroll"):
        transcript = ui.column().classes("transcript-content")

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
        nonlocal current, snapshot, active_dialog, active_citation, pending
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
            send_button.set_enabled(not pending and not sending)
            latest = json.dumps(messages, sort_keys=True, ensure_ascii=False)
            if latest == snapshot:
                return
            # Preserve the reading position; follow new messages only when already at the bottom.
            await ui.run_javascript("""(() => {
                const el = document.querySelector('.transcript-scroll');
                if (el) {
                    const remaining = el.scrollHeight - el.scrollTop - el.clientHeight;
                    el.dataset.follow = String(remaining < 80);
                }
            })()""")
            snapshot = latest
            transcript.clear()
            with transcript:
                if not messages:
                    with ui.column().classes("chat-empty"):
                        ui.icon("menu_book", size="34px").classes("empty-icon")
                        ui.label("Что вы хотите узнать по документам?").classes(
                            "text-2xl font-medium"
                        )
                        ui.label(
                            "Задайте вопрос — ответ будет сопровождаться цитатами источников."
                        ).classes("text-sm muted max-w-md")
                for message in messages:
                    style = (
                        "chat-message-user"
                        if message["role"] == "user"
                        else "chat-message-assistant"
                    )
                    with ui.column().classes(f"chat-message {style}"):
                        if message["role"] != "user":
                            ui.label("Legal AI").classes("font-medium text-sm text-primary")
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
                                        ).props("flat no-caps dense").classes(
                                            "max-w-full text-left"
                                        )
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
            ui.run_javascript("""requestAnimationFrame(() => {
                const el = document.querySelector('.transcript-scroll');
                if (el?.dataset.follow === 'true') el.scrollTop = el.scrollHeight;
            })""")
        except Exception as exc:
            log_failure("chat_refresh", exc)
            ui.notify("Не удалось обновить диалог", type="negative")

    async def send():
        nonlocal current, snapshot, sending
        if sending or pending or not question.value or not question.value.strip():
            return
        sending = True
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
        finally:
            sending = False
            send_button.set_enabled(not pending)

    with ui.column().classes("chat-composer"):
        with ui.row().classes("w-full composer-row"):
            question = (
                ui.textarea(label="Вопрос", placeholder="Задайте вопрос по документам")
                .props("outlined rounded maxlength=1500 autogrow rows=1")
                .classes("flex-1 min-w-0 chat-question")
            )
            question.on(
                "keydown",
                send,
                js_handler="""(event) => {
                    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
                        event.preventDefault();
                        emit();
                    }
                }""",
            )
            send_button = (
                ui.button(icon="arrow_upward", on_click=send)
                .props('round unelevated aria-label="Отправить"')
                .classes("mb-1")
            )
        ui.label("Локальная модель · Проверяйте выводы по источникам").classes(
            "text-xs muted self-center text-center"
        )
    ui.timer(2, refresh, immediate=True)
