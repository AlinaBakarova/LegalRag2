from nicegui import events, run, ui

from legal_workspace.documents.parsers import DocumentError
from legal_workspace.documents.service import MAX_FILE, DocumentService

LABELS = {
    "uploaded": "Ожидает обработки",
    "ready": "Текст извлечён",
    "needs_ocr": "Требуется OCR",
    "needs_review": "Нужна сверка с оригиналом",
    "failed": "Ошибка обработки",
}
ERRORS = {
    "utf8_required": "TXT должен быть сохранён в UTF-8.",
    "empty_document": "Не найден текст.",
    "encrypted_pdf": "PDF защищён паролем.",
    "unsupported_format": "Поддерживаются TXT, PDF и DOCX.",
    "file_size_limit": "Нужен непустой файл размером до 20 МБ.",
}


def document_panel(service: DocumentService):
    groups = ui.column().classes("w-full gap-4")
    last_snapshot = None

    async def show(document_id):
        try:
            document = await run.io_bound(service.get, document_id)
            spans = await run.io_bound(service.spans, document_id)
        except DocumentError:
            ui.notify("Документ больше недоступен", type="warning")
            return
        with ui.dialog() as dialog, ui.card().classes("w-full max-w-4xl"):
            ui.label(document["source"].get("title", document["filename"])).classes("text-xl")
            ui.label(LABELS[document["status"]])
            if document["source"]:
                for field in ("edition", "verified_on", "source_url"):
                    ui.label(f"{field}: {document['source'].get(field, '')}").classes("text-sm")
            for warning in document["warnings"]:
                ui.label(ERRORS.get(warning, "Неполная обработка: " + warning)).classes(
                    "text-amber-900"
                )
            ui.link("Скачать оригинал", f"/api/documents/{document_id}/original", new_tab=True)
            with ui.scroll_area().classes("w-full h-96"):
                if not spans:
                    ui.label("Текст пока недоступен. Проверьте статус документа и запуск worker.")
                for span in spans:
                    ui.label(
                        " · ".join(f"{key}: {value}" for key, value in span["locator"].items())
                    ).classes("text-xs muted")
                    ui.label(span["text"]).classes("whitespace-pre-wrap break-words w-full")
            ui.button("Закрыть", on_click=dialog.close)
        dialog.open()

    async def remove(document_id):
        with ui.dialog() as dialog, ui.card():
            ui.label("Удалить документ и извлечённый текст?")
            ui.label("Восстановить файл можно будет только повторной загрузкой.").classes("text-sm")

            async def confirm():
                try:
                    await run.io_bound(service.delete, document_id)
                    ui.notify("Документ удалён")
                except DocumentError:
                    ui.notify("Не удалось удалить документ", type="negative")
                dialog.close()
                await refresh()

            ui.button("Удалить", on_click=confirm, color="negative")
            ui.button("Отмена", on_click=dialog.close)
        dialog.open()

    async def refresh():
        nonlocal last_snapshot
        documents = await run.io_bound(service.list)
        snapshot = [(d["id"], d["status"]) for d in documents]
        if snapshot == last_snapshot:
            return
        last_snapshot = snapshot
        groups.clear()
        with groups:
            for kind, title in (("base", "Базовая библиотека"), ("user", "Мои документы")):
                ui.label(title).classes("font-medium")
                entries = [d for d in documents if d["kind"] == kind]
                if not entries:
                    ui.label("Документы ещё не добавлены").classes("text-sm muted")
                for document in entries:
                    with ui.column().classes("gap-1 w-full"):
                        ui.button(
                            document["source"].get("title", document["filename"]),
                            on_click=lambda _event, doc_id=document["id"]: show(doc_id),
                        ).props("flat no-caps").classes("w-full break-all")
                        ui.label(LABELS[document["status"]]).classes("text-xs muted")
                        if kind == "user":
                            ui.button(
                                "Удалить документ",
                                on_click=lambda _event, doc_id=document["id"]: remove(doc_id),
                            ).props("flat dense no-caps").classes("text-xs")
                        else:
                            ui.label("Защищённый источник").classes("text-xs muted")
                ui.separator()

    async def upload(event: events.UploadEventArguments):
        try:
            data = await event.file.read()
            await run.io_bound(service.upload, event.file.name, data)
            ui.notify("Документ сохранён и поставлен на обработку", type="positive")
            await refresh()
        except DocumentError as exc:
            ui.notify(ERRORS.get(str(exc), "Не удалось загрузить документ"), type="negative")
        finally:
            uploader.reset()

    uploader = ui.upload(
        on_upload=upload,
        auto_upload=True,
        label="Загрузить документ",
        max_file_size=MAX_FILE,
        on_rejected=lambda: ui.notify("Файл превышает 20 МБ", type="warning"),
    )
    uploader.props('accept=".txt,.pdf,.docx"').classes("w-full")
    ui.label("TXT (UTF-8), PDF, DOCX · до 20 МБ. Для обработки запустите worker.").classes(
        "text-xs muted"
    )
    ui.timer(2, refresh, immediate=True)
