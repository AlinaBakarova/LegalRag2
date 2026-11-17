"""Persistent local chat, citations and source dependencies; search indices unchanged."""

import sqlalchemy as sa
from alembic import op

revision = "0004_chat"
down_revision = "0003_search"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "chat_sessions",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("title", sa.String(80), nullable=False),
        sa.Column("created_at", sa.Float(), nullable=False),
    )
    op.create_table(
        "chat_messages",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column(
            "session_id",
            sa.String(32),
            sa.ForeignKey("chat_sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("role", sa.String(16), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("reply_to", sa.String(32), nullable=True),
        sa.Column("scope", sa.JSON(), nullable=True),
        sa.Column("model", sa.String(120), nullable=True),
        sa.Column("error_code", sa.String(80), nullable=True),
        sa.Column("usage", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.Float(), nullable=False),
        sa.UniqueConstraint("session_id", "ordinal"),
    )
    op.create_index("ix_chat_messages_session_id", "chat_messages", ["session_id"])
    op.create_table(
        "chat_dependencies",
        sa.Column(
            "message_id",
            sa.String(32),
            sa.ForeignKey("chat_messages.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "document_id",
            sa.String(32),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("text_revision", sa.String(32), nullable=False),
    )
    op.create_table(
        "chat_citations",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column(
            "message_id",
            sa.String(32),
            sa.ForeignKey("chat_messages.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "document_id",
            sa.String(32),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("text_revision", sa.String(32), nullable=False),
        sa.Column("span_id", sa.Integer(), nullable=False),
        sa.Column("start", sa.Integer(), nullable=False),
        sa.Column("end", sa.Integer(), nullable=False),
        sa.Column("quote", sa.Text(), nullable=False),
        sa.Column("locator", sa.JSON(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("heading", sa.Text(), nullable=False),
        sa.Column("claim_ordinal", sa.Integer(), nullable=False),
    )
    op.create_index("ix_chat_citations_message_id", "chat_citations", ["message_id"])


def downgrade():
    for table in ("chat_citations", "chat_dependencies", "chat_messages", "chat_sessions"):
        op.drop_table(table)
