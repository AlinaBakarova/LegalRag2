import time

from sqlalchemy import func, select

from legal_workspace.jobs.queue import JobQueue
from legal_workspace.persistence.database import Database
from legal_workspace.persistence.models import Job, WorkerState, Workspace

PUBLIC_JOB_FIELDS = (
    "id",
    "kind",
    "status",
    "attempts",
    "created_at",
    "finished_at",
    "error_code",
    "result",
)


class WorkspaceService:
    def __init__(self, database: Database):
        self.database = database
        self.queue = JobQueue(database)

    def schedule_check(self) -> str:
        return self.queue.enqueue("system.check", priority=10)

    def jobs(self) -> list[dict]:
        with self.database.engine.connect() as connection:
            rows = connection.execute(
                select(*(getattr(Job, field) for field in PUBLIC_JOB_FIELDS))
                .order_by(Job.created_at.desc(), Job.id)
                .limit(20)
            ).mappings()
            return [dict(row) for row in rows]

    def job(self, job_id: str) -> dict | None:
        job = self.queue.get(job_id)
        return {field: job[field] for field in PUBLIC_JOB_FIELDS} if job else None

    def status(self) -> dict:
        cutoff = time.time() - self.database.settings.worker_stale_after_seconds
        with self.database.engine.connect() as connection:
            workspace = connection.execute(select(Workspace.__table__)).mappings().one()
            active = connection.scalar(
                select(func.count())
                .select_from(WorkerState)
                .where(WorkerState.stopped_at.is_(None), WorkerState.heartbeat_at >= cutoff)
            )
            counts = dict(
                connection.execute(select(Job.status, func.count()).group_by(Job.status)).all()
            )
        return {
            "workspace": dict(workspace),
            "worker": {"online": active > 0, "active_count": active},
            "jobs": {
                state: counts.get(state, 0)
                for state in ("queued", "running", "succeeded", "failed")
            },
        }
