"""Persistent workspace, leased jobs and worker presence."""

import sqlalchemy as sa
from alembic import op

revision = "0001_foundation"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "workspace",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("created_at", sa.Float(), nullable=False),
        sa.CheckConstraint("id = 1", name="workspace_singleton"),
    )
    op.create_table(
        "jobs",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("kind", sa.String(80), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("result", sa.JSON(), nullable=True),
        sa.Column("priority", sa.Integer(), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("max_attempts", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.Float(), nullable=False),
        sa.Column("updated_at", sa.Float(), nullable=False),
        sa.Column("available_at", sa.Float(), nullable=False),
        sa.Column("started_at", sa.Float(), nullable=True),
        sa.Column("finished_at", sa.Float(), nullable=True),
        sa.Column("lease_owner", sa.String(32), nullable=True),
        sa.Column("lease_token", sa.String(32), nullable=True),
        sa.Column("lease_expires_at", sa.Float(), nullable=True),
        sa.Column("error_code", sa.String(80), nullable=True),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'succeeded', 'failed')", name="job_status"
        ),
        sa.CheckConstraint("attempts >= 0 AND max_attempts > 0", name="job_attempts"),
        sa.CheckConstraint(
            "status != 'running' OR "
            "(lease_owner IS NOT NULL AND lease_token IS NOT NULL "
            "AND lease_expires_at IS NOT NULL)",
            name="running_job_lease",
        ),
    )
    op.create_index("ix_jobs_claim", "jobs", ["status", "available_at", "priority", "created_at"])
    op.create_table(
        "workers",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("started_at", sa.Float(), nullable=False),
        sa.Column("heartbeat_at", sa.Float(), nullable=False),
        sa.Column("stopped_at", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("workers")
    op.drop_table("jobs")
    op.drop_table("workspace")
