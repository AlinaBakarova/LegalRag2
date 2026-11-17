import sqlalchemy as sa
from sqlalchemy import JSON, CheckConstraint, Float, Index, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Workspace(Base):
    __tablename__ = "workspace"
    __table_args__ = (CheckConstraint("id = 1", name="workspace_singleton"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[float] = mapped_column(Float)


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('queued', 'running', 'succeeded', 'failed')", name="job_status"
        ),
        CheckConstraint("attempts >= 0 AND max_attempts > 0", name="job_attempts"),
        CheckConstraint(
            "status != 'running' OR "
            "(lease_owner IS NOT NULL AND lease_token IS NOT NULL "
            "AND lease_expires_at IS NOT NULL)",
            name="running_job_lease",
        ),
        Index("ix_jobs_claim", "status", "available_at", "priority", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    kind: Mapped[str] = mapped_column(String(80))
    status: Mapped[str] = mapped_column(String(16))
    payload: Mapped[dict] = mapped_column(JSON)
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    priority: Mapped[int] = mapped_column(Integer)
    attempts: Mapped[int] = mapped_column(Integer)
    max_attempts: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[float] = mapped_column(Float)
    updated_at: Mapped[float] = mapped_column(Float)
    available_at: Mapped[float] = mapped_column(Float)
    started_at: Mapped[float | None] = mapped_column(Float, nullable=True)
    finished_at: Mapped[float | None] = mapped_column(Float, nullable=True)
    lease_owner: Mapped[str | None] = mapped_column(String(32), nullable=True)
    lease_token: Mapped[str | None] = mapped_column(String(32), nullable=True)
    lease_expires_at: Mapped[float | None] = mapped_column(Float, nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(80), nullable=True)


class WorkerState(Base):
    __tablename__ = "workers"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    started_at: Mapped[float] = mapped_column(Float)
    heartbeat_at: Mapped[float] = mapped_column(Float)
    stopped_at: Mapped[float | None] = mapped_column(Float, nullable=True)


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        CheckConstraint("kind IN ('base', 'user')", name="document_kind"),
        CheckConstraint(
            "status IN ('uploaded', 'ready', 'needs_ocr', 'needs_review', "
            "'failed', 'deleting', 'deleted')",
            name="document_status",
        ),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    kind: Mapped[str] = mapped_column(String(8))
    filename: Mapped[str] = mapped_column(String(255))
    sha256: Mapped[str] = mapped_column(String(64))
    size: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(20))
    created_at: Mapped[float] = mapped_column(Float)
    parser_version: Mapped[str | None] = mapped_column(String(80), nullable=True)
    warnings: Mapped[list] = mapped_column(JSON)
    source: Mapped[dict] = mapped_column(JSON)
    corpus_key: Mapped[str | None] = mapped_column(String(200), unique=True, nullable=True)


class SourceSpan(Base):
    __tablename__ = "source_spans"
    __table_args__ = (sa.UniqueConstraint("document_id", "ordinal"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[str] = mapped_column(
        sa.ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    ordinal: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(sa.Text)
    locator: Mapped[dict] = mapped_column(JSON)
    start: Mapped[int] = mapped_column(Integer)
    end: Mapped[int] = mapped_column(Integer)
