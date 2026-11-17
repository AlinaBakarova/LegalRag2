import logging
import tempfile
import traceback

from nicegui import events, run, ui

from legal_workspace.documents.parsers import DocumentError
from legal_workspace.documents.service import MAX_FILE, DocumentService

logger = logging.getLogger(__name__)


def log_failure(action, exc):
    frames = [(f.filename, f.lineno, f.name) for f in traceback.extract_tb(exc.__traceback__)]
    logger.warning(
        "document_ui_failed action=%s error_type=%s stack=%s", action, type(exc).__name__, frames
    )


ERRORS = {
    "utf8_required": "Сохраните TXT в кодировке UTF-8 и загрузите заново.",
    "unsupported_format": "Поддерживаются TXT, PDF и DOCX.",
    "file_size_limit": "Выберите непустой файл размером до 25 МБ.",
}


def document_panel(service: DocumentService):
    groups = ui.column().classes("w-full gap-4")
    last_snapshot = None

    def state(document):
        status = document["status"]
        if status == "uploaded":
            with ui.spinner(size="16px", color="primary"):
                ui.tooltip("Документ обрабатывается")
        elif status == "failed":
            ui.label("Не удалось обработать документ").classes("text-xs text-red-800")
        elif status in {"needs_review", "needs_ocr"}:
            ui.label("Часть документа не удалось прочитать").classes("text-xs text-amber-900")

    async def show(document_id):
        try:
            document = await run.io_bound(service.get, document_id)
        except DocumentError:
            ui.notify("Документ больше недоступен", type="warning")
            return
        with ui.dialog() as dialog, ui.card().classes("w-full max-w-2xl"):
            ui.label(document["source"].get("title", document["filename"])).classes(
                "text-xl break-words w-full"
            )
            state(document)
            if document["source"]:
                for field, title in (
                    ("edition", "Редакция"),
                    ("verified_on", "Дата проверки источника"),
                ):
                    ui.label(f"{title}: {document['source'].get(field, '')}").classes(
                        "text-sm break-words w-full"
                    )
                source_url = document["source"].get("source_url", "")
                if source_url.startswith(("https://", "http://")):
                    ui.link("Источник документа", source_url, new_tab=True)
            if document["filename"].lower().endswith(".pdf"):
                ui.link(
                    "Открыть оригинал PDF",
                    f"/api/documents/{document_id}/original?download=false",
                    new_tab=True,
                )
            ui.link("Скачать оригинал", f"/api/documents/{document_id}/original", new_tab=True)
            ui.button("Закрыть", on_click=dialog.close).props("no-caps")
        dialog.open()

    async def remove(document_id):
        with ui.dialog() as dialog, ui.card():
            ui.label("Удалить документ?")
            ui.label("Восстановить файл можно будет только повторной загрузкой.").classes("text-sm")

            async def confirm():
                try:
                    await run.io_bound(service.delete, document_id)
                    ui.notify("Документ удалён")
                except Exception as exc:
                    log_failure("delete", exc)
                    ui.notify("Не удалось удалить документ. Попробуйте ещё раз.", type="negative")
                dialog.close()
                await refresh()

            ui.button("Удалить", on_click=confirm, color="negative").props("no-caps")
            ui.button("Отмена", on_click=dialog.close).props("no-caps")
        dialog.open()

    async def retry(document_id):
        try:
            await run.io_bound(service.reprocess, document_id)
            await refresh()
        except Exception as exc:
            log_failure("reprocess", exc)
            ui.notify("Не удалось начать обработку. Попробуйте ещё раз.", type="negative")

    async def refresh():
        nonlocal last_snapshot
        try:
            documents = await run.io_bound(service.list)
        except Exception as exc:
            log_failure("list", exc)
            ui.notify("Не удалось обновить список документов", type="negative")
            return
        snapshot = [(d["id"], d["status"]) for d in documents]
        if snapshot == last_snapshot:
            return
        last_snapshot = snapshot
        groups.clear()
        with groups:
            for kind, title, icon, hint in (
                (
                    "base",
                    "Базовая библиотека",
                    "lock_outline",
                    "Системные документы — недоступны для редактирования",
                ),
                ("user", "Мои документы", "edit", "Можно добавлять и удалять свои документы"),
            ):
                with ui.row().classes("items-center gap-2 flex-nowrap"):
                    ui.label(title).classes("font-medium")
                    with ui.icon(icon, size="16px").classes("muted").props(f'aria-label="{hint}"'):
                        ui.tooltip(hint).props('anchor="top middle" self="bottom middle"')
                entries = [d for d in documents if d["kind"] == kind]
                if not entries:
                    ui.label("Документы ещё не добавлены").classes("text-sm muted")
                for document in entries:
                    with ui.column().classes("gap-1 w-full"):
                        with ui.row().classes("items-center flex-nowrap w-full gap-1"):
                            ui.button(
                                document["source"].get("title", document["filename"]),
                                on_click=lambda _event, doc_id=document["id"]: show(doc_id),
                            ).props("flat no-caps").classes("min-w-0 flex-1 document-name")
                            if kind == "user":
                                with (
                                    ui.button(
                                        icon="delete_outline",
                                        on_click=lambda _event, doc_id=document["id"]: remove(
                                            doc_id
                                        ),
                                    )
                                    .props('flat round dense aria-label="Удалить документ"')
                                    .classes("muted")
                                ):
                                    ui.tooltip("Удалить документ").props(
                                        'anchor="top middle" self="bottom middle"'
                                    )
                        state(document)
                        if kind == "user" and document["status"] in {
                            "failed",
                            "needs_review",
                            "needs_ocr",
                        }:
                            with (
                                ui.button(
                                    icon="refresh",
                                    on_click=lambda _event, doc_id=document["id"]: retry(doc_id),
                                )
                                .props('flat round dense aria-label="Повторить обработку"')
                                .classes("muted")
                            ):
                                ui.tooltip("Повторить обработку")
                ui.separator()

    async def upload(event: events.UploadEventArguments):
        try:
            # NiceGUI has already spooled the upload. Read it incrementally into storage.
            with tempfile.TemporaryFile() as stream:
                async for chunk in event.file.iterate():
                    await run.io_bound(stream.write, chunk)
                stream.seek(0)
                await run.io_bound(service.upload_stream, event.file.name, stream)
            await refresh()
        except DocumentError as exc:
            logger.warning("document_upload_rejected code=%s", str(exc))
            ui.notify(ERRORS.get(str(exc), "Не удалось обработать документ"), type="negative")
        except Exception as exc:
            log_failure("upload", exc)
            ui.notify("Не удалось загрузить документ. Попробуйте ещё раз.", type="negative")
        finally:
            uploader.reset()

    uploader = ui.upload(
        on_upload=upload,
        auto_upload=True,
        label="Загрузить документ",
        max_file_size=MAX_FILE,
        on_rejected=lambda: ui.notify("Файл должен быть не больше 25 МБ", type="warning"),
    )
    uploader.props('accept=".txt,.pdf,.docx"').classes("w-full")
    ui.label("TXT (UTF-8), PDF, DOCX · до 25 МБ · PDF до 300 страниц").classes("text-xs muted")
    ui.timer(2, refresh, immediate=True)
