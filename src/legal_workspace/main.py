import argparse
import logging
import signal
import sys

from pydantic import ValidationError

from legal_workspace.config import Settings
from legal_workspace.persistence.database import Database


def create_app(settings: Settings):
    # Keep NiceGUI's process-global page registry out of API-only tests and the worker.
    from nicegui import ui

    from legal_workspace.api import create_api
    from legal_workspace.ui.layout import register_pages

    app = create_api(settings)
    register_pages(app.state.service)
    ui.run_with(
        app, title="Legal AI Workspace", dark=False, language="ru", show_welcome_message=False
    )
    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="Local Legal AI Workspace")
    parser.add_argument(
        "command",
        choices=[
            "init",
            "serve",
            "worker",
            "check",
            "import-corpus",
            "reprocess-documents",
            "prepare-model",
            "index-documents",
            "search-status",
            "evaluate-retrieval",
            "diagnose-retrieval",
        ],
    )
    parser.add_argument("--manifest", default="corpus/manifest.json")
    parser.add_argument("--query")
    parser.add_argument("--chunk-id", type=int)
    parser.add_argument("--failed-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    try:
        settings = Settings()
    except ValidationError as exc:
        fields = ", ".join(".".join(map(str, error["loc"])) or "timing" for error in exc.errors())
        print(f"Invalid configuration fields: {fields}. See .env.example.", file=sys.stderr)
        return 2

    try:
        if args.command == "diagnose-retrieval":
            import json

            from legal_workspace.retrieval.diagnostics import diagnose

            if not args.query or args.chunk_id is None:
                parser.error("diagnose-retrieval requires --query and --chunk-id")
            print(
                json.dumps(
                    diagnose(settings, args.query, args.chunk_id), ensure_ascii=False, indent=2
                )
            )
            return 0
        if args.command == "serve":
            import uvicorn

            uvicorn.run(
                create_app(settings), host=settings.host, port=settings.port, access_log=False
            )
            return 0

        database = Database(settings)
        try:
            database.initialize()
            if args.command == "init":
                print("Workspace initialized; migrations are up to date.")
            elif args.command == "check":
                from legal_workspace.services import WorkspaceService

                print(WorkspaceService(database).schedule_check())
            elif args.command == "import-corpus":
                from pathlib import Path

                from legal_workspace.documents.corpus_import import import_corpus
                from legal_workspace.documents.service import DocumentService

                ids = import_corpus(DocumentService(database), Path(args.manifest))
                print(f"Corpus documents imported: {len(ids)}")
            elif args.command == "reprocess-documents":
                from legal_workspace.documents.service import DocumentService

                documents = DocumentService(database)
                count = 0
                for document in documents.list():
                    if args.failed_only and document["status"] != "failed":
                        continue
                    documents.reprocess(document["id"])
                    count += 1
                print(f"Documents queued for reprocessing: {count}")
            elif args.command == "prepare-model":
                from legal_workspace.retrieval.embeddings import prepare_model

                manifest = prepare_model(settings.data_dir)
                print(f"Local model prepared: {manifest['model']} @ {manifest['revision']}")
            elif args.command == "index-documents":
                from legal_workspace.retrieval.service import SearchService

                count = SearchService(database).schedule_missing(args.force)
                print(f"Documents queued for indexing: {count}")
            elif args.command == "search-status":
                import json

                from legal_workspace.retrieval.service import SearchService
                from legal_workspace.services import WorkspaceService

                workspace = WorkspaceService(database)
                print(
                    json.dumps(
                        {
                            "search": SearchService(database).status(),
                            "worker": workspace.status()["worker"],
                            "jobs": workspace.jobs(),
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
            elif args.command == "evaluate-retrieval":
                from legal_workspace.retrieval.evaluation import evaluate

                print(evaluate(settings.data_dir))
            elif args.command == "worker":
                from legal_workspace.jobs.worker import Worker

                worker = Worker(database, settings)
                signal.signal(signal.SIGTERM, lambda *_: worker.stop())
                signal.signal(signal.SIGINT, lambda *_: worker.stop())
                from legal_workspace.rag.trace import rag_log_file

                with rag_log_file(settings.data_dir):
                    worker.run()
        finally:
            database.close()
    except Exception as exc:
        from legal_workspace.retrieval.embeddings import SearchError

        if isinstance(exc, SearchError):
            print(
                f"Search unavailable: {exc}. Run prepare-model, then index-documents.",
                file=sys.stderr,
            )
            return 1
        # Log only the class: SQL/HTTP exceptions can embed sensitive input.
        logging.getLogger(__name__).error(
            "startup_or_runtime_failed error_type=%s", type(exc).__name__
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
