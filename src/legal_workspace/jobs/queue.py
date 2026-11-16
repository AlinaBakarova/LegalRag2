import time
from collections.abc import Callable
from dataclasses import dataclass
from uuid import uuid4

from sqlalchemy import select, update

from legal_workspace.persistence.database import Database
from legal_workspace.persistence.models import Job


@dataclass(frozen=True)
class Claim:
    id: str
    kind: str
    payload: dict
    token: str


class JobQueue:
    """At-least-once delivery. Every handler must tolerate a repeated execution."""

    def __init__(self, database: Database, clock: Callable[[], float] = time.time):
        self.database = database
        self.clock = clock

    def enqueue(
        self, kind: str, payload: dict | None = None, *, priority: int = 0, max_attempts: int = 3
    ) -> str:
        if not kind or max_attempts < 1:
            raise ValueError("A kind and a positive retry limit are required")
        now = self.clock()
        job_id = uuid4().hex
        with self.database.transaction() as connection:
            connection.execute(
                Job.__table__.insert().values(
                    id=job_id,
                    kind=kind,
                    status="queued",
                    payload=payload or {},
                    priority=priority,
                    attempts=0,
                    max_attempts=max_attempts,
                    created_at=now,
                    updated_at=now,
                    available_at=now,
                )
            )
        return job_id

    def get(self, job_id: str) -> dict | None:
        with self.database.engine.connect() as connection:
            row = (
                connection.execute(select(Job.__table__).where(Job.id == job_id)).mappings().first()
            )
            return dict(row) if row else None

    def claim(self, owner: str, lease_seconds: float) -> Claim | None:
        if lease_seconds <= 0:
            raise ValueError("Lease duration must be positive")
        cleared = {"lease_owner": None, "lease_token": None, "lease_expires_at": None}
        # Serialize recovery + selection + claim, even with competing worker processes.
        with self.database.transaction(immediate=True) as connection:
            now = self.clock()
            expired = (Job.status == "running") & (Job.lease_expires_at <= now)
            connection.execute(
                update(Job)
                .where(expired, Job.attempts >= Job.max_attempts)
                .values(
                    **cleared,
                    status="failed",
                    error_code="lease_expired",
                    updated_at=now,
                    finished_at=now,
                )
            )
            connection.execute(
                update(Job)
                .where(expired, Job.attempts < Job.max_attempts)
                .values(
                    **cleared,
                    status="queued",
                    error_code="lease_expired",
                    updated_at=now,
                    available_at=now,
                )
            )
            row = (
                connection.execute(
                    select(Job.__table__)
                    .where(Job.status == "queued", Job.available_at <= now)
                    .order_by(Job.priority.desc(), Job.created_at, Job.id)
                    .limit(1)
                )
                .mappings()
                .first()
            )
            if row is None:
                return None
            token = uuid4().hex
            connection.execute(
                update(Job)
                .where(Job.id == row["id"])
                .values(
                    status="running",
                    attempts=Job.attempts + 1,
                    updated_at=now,
                    started_at=now,
                    lease_owner=owner,
                    lease_token=token,
                    lease_expires_at=now + lease_seconds,
                    error_code=None,
                )
            )
            return Claim(row["id"], row["kind"], row["payload"], token)

    def _owned(self, claim: Claim, now: float):
        return (
            (Job.id == claim.id)
            & (Job.status == "running")
            & (Job.lease_token == claim.token)
            & (Job.lease_expires_at > now)
        )

    def renew(self, claim: Claim, lease_seconds: float) -> bool:
        if lease_seconds <= 0:
            raise ValueError("Lease duration must be positive")
        with self.database.transaction(immediate=True) as connection:
            now = self.clock()
            result = connection.execute(
                update(Job)
                .where(self._owned(claim, now))
                .values(
                    lease_expires_at=now + lease_seconds,
                    updated_at=now,
                )
            )
            return result.rowcount == 1

    def finish(self, claim: Claim, result: dict) -> bool:
        with self.database.transaction(immediate=True) as connection:
            now = self.clock()
            changed = connection.execute(
                update(Job)
                .where(self._owned(claim, now))
                .values(
                    status="succeeded",
                    result=result,
                    finished_at=now,
                    updated_at=now,
                    lease_owner=None,
                    lease_token=None,
                    lease_expires_at=None,
                )
            )
            return changed.rowcount == 1

    def fail(self, claim: Claim, error_code: str, *, retryable: bool = True) -> bool:
        with self.database.transaction(immediate=True) as connection:
            now = self.clock()
            row = connection.execute(
                select(Job.attempts, Job.max_attempts).where(self._owned(claim, now))
            ).first()
            if row is None:
                return False
            retry = retryable and row.attempts < row.max_attempts
            connection.execute(
                update(Job)
                .where(self._owned(claim, now))
                .values(
                    status="queued" if retry else "failed",
                    error_code=error_code,
                    available_at=now + min(2**row.attempts, 60),
                    updated_at=now,
                    finished_at=None if retry else now,
                    lease_owner=None,
                    lease_token=None,
                    lease_expires_at=None,
                )
            )
            return True
