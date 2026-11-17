"""Atomic search generations and SQLite FTS5."""

import sqlalchemy as sa
from alembic import op

revision = "0003_search"
down_revision = "0002_documents"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("documents", sa.Column("text_revision", sa.String(32), nullable=True))
    op.add_column("documents", sa.Column("active_index_id", sa.String(32), nullable=True))
    op.execute(
        "UPDATE documents SET text_revision=lower(hex(randomblob(16))) "
        "WHERE status IN ('ready','needs_review')"
    )
    op.create_table(
        "index_generations",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column(
            "document_id",
            sa.String(32),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("text_revision", sa.String(32), nullable=False),
        sa.Column("fingerprint", sa.String(64), nullable=False),
        sa.Column("dimension", sa.Integer(), nullable=False),
        sa.Column("state", sa.String(16), nullable=False),
        sa.Column("job_id", sa.String(32), nullable=False),
        sa.Column("lease_token", sa.String(32), nullable=False),
        sa.Column("created_at", sa.Float(), nullable=False),
    )
    op.create_index("ix_index_generations_document_id", "index_generations", ["document_id"])
    op.create_table(
        "chunks",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "generation_id",
            sa.String(32),
            sa.ForeignKey("index_generations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "span_id",
            sa.Integer(),
            sa.ForeignKey("source_spans.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("start", sa.Integer(), nullable=False),
        sa.Column("end", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("heading", sa.Text(), nullable=False),
        sa.Column("terms", sa.Text(), nullable=False),
        sa.Column("embedding", sa.LargeBinary(), nullable=False),
        sa.UniqueConstraint("generation_id", "ordinal"),
        sqlite_autoincrement=True,
    )
    op.create_index("ix_chunks_generation_id", "chunks", ["generation_id"])
    op.execute(
        "CREATE VIRTUAL TABLE chunk_fts USING fts5(terms, content='chunks', content_rowid='id')"
    )
    op.execute(
        "CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN "
        "INSERT INTO chunk_fts(rowid,terms) VALUES(new.id,new.terms); END"
    )
    op.execute(
        "CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN "
        "INSERT INTO chunk_fts(chunk_fts,rowid,terms) VALUES('delete',old.id,old.terms); END"
    )
    op.execute(
        "CREATE TRIGGER chunks_au AFTER UPDATE OF terms ON chunks BEGIN "
        "INSERT INTO chunk_fts(chunk_fts,rowid,terms) VALUES('delete',old.id,old.terms); "
        "INSERT INTO chunk_fts(rowid,terms) VALUES(new.id,new.terms); END"
    )


def downgrade():
    for name in ("chunks_ai", "chunks_ad", "chunks_au"):
        op.execute(f"DROP TRIGGER {name}")
    op.execute("DROP TABLE chunk_fts")
    op.drop_table("chunks")
    op.drop_table("index_generations")
    op.drop_column("documents", "active_index_id")
    op.drop_column("documents", "text_revision")
