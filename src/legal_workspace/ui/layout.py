from datetime import datetime
from zoneinfo import ZoneInfo

from nicegui import run, ui

from legal_workspace.services import WorkspaceService

STATUS_LABELS = {
    "queued": "Ожидает запуска",
    "running": "Выполняется",
    "succeeded": "Проверка пройдена",
    "failed": "Не удалось выполнить",
}


def register_pages(service: WorkspaceService) -> None:
    @ui.page("/")
    def workspace_page():
        ui.colors(primary="#264e45", secondary="#94734b", accent="#264e45")
        ui.add_css("""
            body { background: #f4f5f2; color: #26332e; font-family: system-ui, sans-serif; }
            .nicegui-content { padding: 0; gap: 0; }
            .workspace-header { background: #fff; color: #26332e;
                border-bottom: 1px solid #dde3dd; }
            .workspace-grid { display: grid; grid-template-columns: 270px minmax(300px, 1fr) 280px;
                gap: 24px; padding: 28px; width: 100%; max-width: 1600px; margin: 0 auto; }
            .panel { background: #fff; border: 1px solid #dfe5de; border-radius: 16px;
                padding: 24px; box-shadow: none; width: 100%; }
            .muted { color: #68766e; }
            .eyebrow { font-size: 11px; letter-spacing: .14em; font-weight: 650; }
            .chat-panel { min-height: 620px; }
            .empty-icon { background: #eef3ed; color: #476548; border-radius: 20px; padding: 18px; }
            @media (max-width: 1100px) {
                .workspace-grid { grid-template-columns: 230px minmax(280px, 1fr); }
                .status-panel { grid-column: 1 / -1; }
            }
            @media (max-width: 700px) {
                .workspace-grid { grid-template-columns: minmax(0, 1fr); padding: 16px; }
                .chat-panel { min-height: 420px; }
            }
        """)

        with ui.header().classes("workspace-header items-center px-7 py-4"):
            ui.icon("balance", size="28px").classes("text-primary")
            with ui.column().classes("gap-0"):
                ui.label("LEGAL / WORKSPACE").classes("text-base font-semibold tracking-wide")
                ui.label("Пространство для работы с правовыми документами").classes("text-xs muted")
            ui.space()
            ui.badge("Локальное приложение", color="primary").props("outline")

        with ui.element("main").classes("workspace-grid"):
            with ui.column().classes("panel gap-6"):
                ui.label("ДОКУМЕНТЫ").classes("eyebrow muted")
                with ui.column().classes("gap-3 w-full"):
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("account_balance", size="20px")
                        ui.label("Базовая библиотека").classes("font-medium")
                        ui.icon("lock", size="14px").classes("muted")
                    ui.label("Нормативные документы ещё не добавлены.").classes("text-sm muted")
                ui.separator()
                with ui.column().classes("gap-3 w-full"):
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("folder_open", size="20px")
                        ui.label("Мои документы").classes("font-medium")
                    ui.label("Здесь будут ваши файлы и результаты их обработки.").classes(
                        "text-sm muted"
                    )
                    ui.button("Загрузить документ", icon="add").props("outline no-caps").disable()
                ui.space()
                ui.label("Загрузка и просмотр документов появятся на следующем этапе.").classes(
                    "text-xs muted"
                )

            with ui.column().classes("panel chat-panel gap-5"):
                with ui.row().classes("w-full items-center"):
                    ui.label("Рабочее пространство").classes("text-lg font-semibold")
                    ui.space()
                    ui.badge("Этап 1", color="secondary").props("outline")
                ui.separator()
                with ui.column().classes("flex-1 w-full items-center justify-center gap-5 py-12"):
                    ui.icon("auto_stories", size="38px").classes("empty-icon")
                    ui.label("Основа для внимательной работы с документами").classes(
                        "text-2xl font-medium text-center max-w-md"
                    )
                    ui.label(
                        "Библиотека, вопросы и проверяемые источники — в одном месте. "
                        "Сейчас доступна проверка сохранения заданий и работы приложения."
                    ).classes("text-center text-sm muted max-w-md leading-relaxed")
                with ui.row().classes("w-full items-center gap-3"):
                    ui.input(placeholder="Чат будет доступен после подключения поиска и LLM").props(
                        "outlined aria-label=Вопрос"
                    ).classes("flex-1").disable()
                    ui.button(icon="arrow_upward").props("round aria-label=Отправить").disable()
                ui.label("Документы и вопросы пока не передаются внешним AI-сервисам.").classes(
                    "text-xs muted"
                )

            with ui.column().classes("panel status-panel gap-4"):
                ui.label("СОСТОЯНИЕ ПРИЛОЖЕНИЯ").classes("eyebrow muted")
                worker_label = ui.label("Проверяем рабочий процесс…").classes("font-medium")
                worker_help = ui.label().classes("text-sm muted")
                counters = ui.label().classes("text-sm muted")
                ui.separator()
                ui.label("Проверка готовности").classes("font-medium")
                ui.label(
                    "Проверочное задание сохраняется и выполняется в фоне. "
                    "Результат останется доступен после перезапуска."
                ).classes("text-sm muted")

                async def enqueue_check():
                    check_button.disable()
                    try:
                        await run.io_bound(service.schedule_check)
                        ui.notify("Проверка поставлена в очередь", type="positive")
                        await refresh_status()
                    except Exception:
                        ui.notify(
                            "Не удалось сохранить задание. Попробуйте ещё раз.", type="negative"
                        )
                    finally:
                        check_button.enable()

                check_button = (
                    ui.button("Проверить рабочий процесс", on_click=enqueue_check)
                    .props("no-caps")
                    .classes("w-full")
                )
                ui.label("ПОСЛЕДНИЕ ПРОВЕРКИ").classes("eyebrow muted mt-3")
                recent_jobs = ui.column().classes("w-full gap-3")

                async def refresh_status():
                    try:
                        status = await run.io_bound(service.status)
                        jobs = await run.io_bound(service.jobs)
                    except Exception:
                        worker_label.set_text("Состояние временно недоступно")
                        worker_help.set_text("Проверьте запущенные процессы приложения.")
                        return
                    online = status["worker"]["online"]
                    worker_label.set_text(
                        "Рабочий процесс подключён" if online else "Рабочий процесс отключён"
                    )
                    worker_help.set_text(
                        "Фоновые задания готовы к выполнению."
                        if online
                        else "Задания сохраняются в очереди и выполнятся после запуска worker."
                    )
                    counts = status["jobs"]
                    counters.set_text(
                        f"В очереди: {counts['queued']} · Выполнено: {counts['succeeded']}"
                    )
                    recent_jobs.clear()
                    with recent_jobs:
                        if not jobs:
                            ui.label("Проверок пока нет").classes("text-sm muted")
                        for job in jobs[:5]:
                            with ui.column().classes("gap-0"):
                                ui.label(STATUS_LABELS[job["status"]]).classes("text-sm")
                                date = datetime.fromtimestamp(
                                    job["created_at"], ZoneInfo("Europe/Moscow")
                                )
                                ui.label(f"{date:%d.%m.%Y %H:%M:%S} МСК").classes("text-xs muted")

                ui.timer(2.0, refresh_status, immediate=True)
