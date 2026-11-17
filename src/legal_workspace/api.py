import logging
from contextlib import asynccontextmanager
from urllib.parse import quote, urlsplit

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from sqlalchemy.exc import SQLAlchemyError
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from legal_workspace.config import Settings
from legal_workspace.documents.parsers import DocumentError
from legal_workspace.documents.service import MAX_FILE, DocumentService
from legal_workspace.persistence.database import Database
from legal_workspace.services import WorkspaceService

logger = logging.getLogger(__name__)


def create_api(settings: Settings) -> FastAPI:
    database = Database(settings)
    database.initialize()
    service = WorkspaceService(database)

    @asynccontextmanager
    async def lifespan(_app):
        yield
        database.close()

    app = FastAPI(
        title="Legal AI Workspace",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )
    app.state.database = database
    app.state.service = service
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]"])

    @app.middleware("http")
    async def same_origin_mutations(request: Request, call_next):
        origin = request.headers.get("origin")
        if request.method not in {"GET", "HEAD", "OPTIONS"} and origin:
            expected = f"{request.url.scheme}://{request.headers.get('host', '')}"
            parsed = urlsplit(origin)
            if f"{parsed.scheme}://{parsed.netloc}" != expected:
                return JSONResponse({"detail": "Cross-origin request rejected"}, status_code=403)
        return await call_next(request)

    @app.exception_handler(SQLAlchemyError)
    async def database_error(_request: Request, exc: SQLAlchemyError):
        logger.warning("database_request_failed error_type=%s", type(exc).__name__)
        return JSONResponse({"detail": "Storage temporarily unavailable"}, status_code=503)

    @app.get("/api/health")
    def health():
        # This checks web/storage availability. Worker presence is reported separately.
        service.status()
        return {"status": "ok", "database": "ok"}

    @app.get("/api/status")
    def status():
        return service.status()

    @app.get("/api/jobs")
    def jobs():
        return service.jobs()

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str):
        result = service.job(job_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return result

    @app.post("/api/jobs/check", status_code=202)
    def schedule_check():
        return {"id": service.schedule_check()}

    documents = DocumentService(database)

    @app.exception_handler(DocumentError)
    async def document_error(_request: Request, exc: DocumentError):
        code = str(exc)
        status_code = (
            404 if code == "not_found" else 403 if code == "base_document_protected" else 400
        )
        return JSONResponse({"detail": code}, status_code=status_code)

    @app.get("/api/documents")
    def list_documents():
        return documents.list()

    @app.post("/api/documents", status_code=202)
    async def upload_document(file: UploadFile):
        try:
            data = await file.read(MAX_FILE + 1)
            return {"id": await run_in_threadpool(documents.upload, file.filename or "", data)}
        finally:
            await file.close()

    @app.get("/api/documents/{document_id}")
    def get_document(document_id: str):
        return documents.get(document_id)

    @app.get("/api/documents/{document_id}/spans")
    def get_spans(document_id: str):
        return documents.spans(document_id)

    @app.get("/api/documents/{document_id}/original")
    def original(document_id: str):
        with documents.lock:
            document = documents.get(document_id)
            data = documents.path(document_id).read_bytes()
        return Response(
            data,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": "attachment; filename*=UTF-8''"
                + quote(document["filename"]),
                "X-Content-Type-Options": "nosniff",
            },
        )

    @app.delete("/api/documents/{document_id}", status_code=204)
    def delete_document(document_id: str):
        documents.delete(document_id)

    return app
