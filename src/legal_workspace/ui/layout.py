from nicegui import ui

from legal_workspace.services import WorkspaceService


def register_pages(service: WorkspaceService) -> None:
    @ui.page("/")
    def workspace_page():
        ui.colors(primary="#264e45", secondary="#94734b", accent="#264e45")
        ui.add_css("""
            body { background: #f4f5f2; color: #26332e; font-family: system-ui, sans-serif; }
            .nicegui-content { padding: 0; gap: 0; }
            .workspace-header { background: #fff; color: #26332e;
                border-bottom: 1px solid #dde3dd; }
            .workspace-grid { display: grid; grid-template-columns: 280px minmax(0, 1fr);
                gap: 24px; padding: 28px; width: 100%; max-width: 1600px; margin: 0 auto; }
            .panel { background: #fff; border: 1px solid #dfe5de; border-radius: 16px;
                padding: 24px; box-shadow: none; width: 100%; }
            .muted { color: #68766e; }
            .eyebrow { font-size: 11px; letter-spacing: .14em; font-weight: 650; }
            .chat-panel { min-height: 620px; }
            .document-name .q-btn__content { overflow-wrap: anywhere; text-align: left; }
            .empty-icon { background: #eef3ed; color: #476548; border-radius: 20px; padding: 18px; }
            @media (max-width: 1100px) {
                .workspace-grid { grid-template-columns: 230px minmax(0, 1fr); }
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
                from legal_workspace.documents.service import DocumentService
                from legal_workspace.ui.documents import document_panel

                document_panel(DocumentService(service.database))

            with ui.column().classes("panel chat-panel gap-5"):
                with ui.row().classes("w-full items-center"):
                    ui.label("Рабочее пространство").classes("text-lg font-semibold")
                    ui.space()
                    ui.badge("Этап 3", color="secondary").props("outline")
                ui.separator()
                from legal_workspace.ui.search import search_panel

                search_panel(service.database)
                with ui.row().classes("w-full items-center gap-3"):
                    ui.input(placeholder="Чат будет доступен после подключения поиска и LLM").props(
                        "outlined aria-label=Вопрос"
                    ).classes("flex-1").disable()
                    ui.button(icon="arrow_upward").props("round aria-label=Отправить").disable()
                ui.label("Документы и вопросы пока не передаются внешним AI-сервисам.").classes(
                    "text-xs muted"
                )
