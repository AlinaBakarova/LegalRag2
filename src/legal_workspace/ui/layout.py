from nicegui import ui

from legal_workspace.services import WorkspaceService


def register_pages(service: WorkspaceService) -> None:
    @ui.page("/")
    def workspace_page():
        ui.colors(primary="#264e45", secondary="#94734b", accent="#264e45")
        ui.add_css("""
            body { background: #f4f5f2; color: #26332e; font-family: system-ui, sans-serif;
                overflow: hidden; }
            .q-page { min-height: 0 !important; }
            .nicegui-content { padding: 0; gap: 0; height: calc(100dvh - 72px); }
            .workspace-header { background: #fff; color: #26332e; height: 72px;
                border-bottom: 1px solid #dde3dd; }
            .workspace-grid { display: grid; grid-template-columns: 280px minmax(0, 1fr);
                gap: 24px; padding: 24px; width: 100%; max-width: 1600px; margin: 0 auto;
                height: 100%; min-height: 0; }
            .panel { background: #fff; border: 1px solid #dfe5de; border-radius: 16px;
                padding: 24px; box-shadow: none; width: 100%; min-width: 0; min-height: 0; }
            .document-panel { overflow-y: auto; flex-wrap: nowrap; }
            .muted { color: #68766e; }
            .eyebrow { font-size: 11px; letter-spacing: .14em; font-weight: 650; }
            .chat-panel { overflow: hidden; flex-wrap: nowrap; gap: 0; padding: 0; }
            .chat-toolbar { flex: 0 0 auto; padding: 16px 20px; border-bottom: 1px solid #edf0eb;
                flex-wrap: nowrap; gap: 4px; }
            .chat-toolbar-title { flex: 1; min-width: 0; }
            .transcript-scroll { flex: 1; min-height: 0; width: 100%; overflow-y: auto;
                overscroll-behavior: contain; }
            .transcript-content { min-height: 100%; width: 100%; max-width: 850px;
                margin: 0 auto; padding: 24px; gap: 24px; flex-wrap: nowrap; }
            .chat-empty { flex: 1; width: 100%; justify-content: center; align-items: center;
                text-align: center; padding: 24px 0; gap: 18px; }
            .chat-message { width: 100%; gap: 10px; flex-shrink: 0; }
            .chat-message-user { width: fit-content; max-width: 85%; align-self: flex-end;
                border-radius: 18px 18px 4px 18px; background: #eef3ed; padding: 12px 18px; }
            .chat-message-assistant { padding: 4px 0; }
            .chat-message .q-btn__content { overflow-wrap: anywhere; }
            .chat-composer { flex: 0 0 auto; width: 100%; max-width: 898px;
                align-self: center; padding: 12px 24px 16px; gap: 8px; flex-wrap: nowrap; }
            .composer-row { flex-wrap: nowrap; align-items: flex-end; gap: 10px; }
            .chat-question textarea { max-height: 160px; overflow-y: auto !important; }
            .document-name .q-btn__content { overflow-wrap: anywhere; text-align: left; }
            .empty-icon { background: #eef3ed; color: #476548; border-radius: 20px; padding: 18px; }
            .mobile-documents, .documents-backdrop { display: none; }
            @media (max-width: 1100px) {
                .workspace-grid { grid-template-columns: 230px minmax(0, 1fr);
                    gap: 16px; padding: 16px; }
            }
            @media (max-width: 700px) {
                .workspace-header { padding: 12px; gap: 8px; }
                .workspace-subtitle, .local-badge { display: none; }
                .mobile-documents { display: inline-flex; }
                .workspace-grid { grid-template-columns: minmax(0, 1fr); padding: 8px; }
                .document-panel { display: none; position: fixed; z-index: 3000;
                    top: 80px; bottom: 8px; left: 8px; width: min(320px, calc(100vw - 16px)); }
                .document-panel.documents-open { display: flex; }
                .documents-backdrop.documents-open { display: block; position: fixed;
                    inset: 72px 0 0; z-index: 2999; background: #0005; }
                .chat-toolbar { padding: 12px; }
                .chat-toolbar-title { font-size: 15px; }
                .transcript-content { padding: 16px; }
                .chat-composer { padding: 8px 12px 12px; }
            }
        """)

        def toggle_documents():
            sidebar.classes(toggle="documents-open")
            backdrop.classes(toggle="documents-open")

        with ui.header().classes("workspace-header items-center px-7"):
            ui.button(icon="folder_open", on_click=toggle_documents).props(
                'flat round dense aria-label="Документы"'
            ).classes("mobile-documents")
            ui.icon("balance", size="28px").classes("text-primary")
            with ui.column().classes("gap-0"):
                ui.label("LEGAL / WORKSPACE").classes("text-base font-semibold tracking-wide")
                ui.label("Пространство для работы с правовыми документами").classes(
                    "text-xs muted workspace-subtitle"
                )
            ui.space()
            ui.badge("Локальное приложение", color="primary").props("outline").classes(
                "local-badge"
            )

        backdrop = ui.element("div").classes("documents-backdrop").on("click", toggle_documents)
        with ui.element("main").classes("workspace-grid"):
            with ui.column().classes("panel document-panel gap-6") as sidebar:
                with ui.row().classes("w-full items-center justify-between"):
                    ui.label("ДОКУМЕНТЫ").classes("eyebrow muted")
                    ui.button(icon="close", on_click=toggle_documents).props(
                        'flat round dense aria-label="Закрыть документы"'
                    ).classes("mobile-documents")
                from legal_workspace.documents.service import DocumentService
                from legal_workspace.ui.documents import document_panel

                document_panel(DocumentService(service.database))

            with ui.column().classes("panel chat-panel"):
                from legal_workspace.ui.chat import chat_panel

                chat_panel(service.database)
