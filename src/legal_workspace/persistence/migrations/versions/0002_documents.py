"""Original documents and source locations."""

import sqlalchemy as sa
from alembic import op

revision = "0002_documents"
down_revision = "0001_foundation"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "documents",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("kind", sa.String(8), nullable=False),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("size", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("created_at", sa.Float(), nullable=False),
        sa.Column("parser_version", sa.String(80), nullable=True),
        sa.Column("warnings", sa.JSON(), nullable=False),
        sa.Column("source", sa.JSON(), nullable=False),
        sa.Column("corpus_key", sa.String(200), unique=True, nullable=True),
        sa.CheckConstraint("kind IN ('base', 'user')", name="document_kind"),
        sa.CheckConstraint(
            "status IN ('uploaded', 'ready', 'needs_ocr', 'needs_review', "
            "'failed', 'deleting', 'deleted')",
            name="document_status",
        ),
    )
    op.create_table(
        "source_spans",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "document_id",
            sa.String(32),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("locator", sa.JSON(), nullable=False),
        sa.Column("start", sa.Integer(), nullable=False),
        sa.Column("end", sa.Integer(), nullable=False),
        sa.UniqueConstraint("document_id", "ordinal"),
    )
    op.create_index("ix_source_spans_document_id", "source_spans", ["document_id"])


def downgrade():
    op.drop_table("source_spans")
    op.drop_table("documents")
