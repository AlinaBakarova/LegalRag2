import logging
import threading
import time
from collections.abc import Callable
from uuid import uuid4

from sqlalchemy import update

from legal_workspace.config import Settings
from legal_workspace.jobs.queue import Claim, JobQueue
from legal_workspace.persistence.database import Database
from legal_workspace.persistence.models import WorkerState

logger = logging.getLogger(__name__)
Handler = Callable[[dict], dict]


class Worker:
    def __init__(
        self, database: Database, settings: Settings, *, handlers: dict[str, Handler] | None = None
    ):
        self.database = database
        self.settings = settings
        self.queue = JobQueue(database)
        self.id = uuid4().hex
        self.handlers = handlers if handlers is not None else {"system.check": self._check}
        self.stop_requested = threading.Event()
        self._heartbeat_stop = threading.Event()
        self._lock = threading.Lock()
        self._current: Claim | None = None

    def _check(self, _payload: dict) -> dict:
        if not self.database.check():
            raise RuntimeError("Storage integrity check failed")
        return {"storage": "ok"}

    def stop(self) -> None:
        self.stop_requested.set()

    def _heartbeat(self) -> None:
        with self.database.transaction() as connection:
            connection.execute(
                update(WorkerState)
                .where(WorkerState.id == self.id)
                .values(heartbeat_at=time.time())
            )
        with self._lock:
            if self._current is not None:
                self.queue.renew(self._current, self.settings.lease_seconds)

    def _heartbeat_loop(self) -> None:
        while not self._heartbeat_stop.wait(self.settings.heartbeat_interval_seconds):
            try:
                self._heartbeat()
            except Exception as exc:
                # Exception messages and tracebacks can contain future document/API payloads.
                logger.warning("worker_heartbeat_failed error_type=%s", type(exc).__name__)

    def run_once(self) -> bool:
        claim = self.queue.claim(self.id, self.settings.lease_seconds)
        if claim is None:
            return False
        with self._lock:
            self._current = claim
        try:
            handler = self.handlers.get(claim.kind)
            if claim.kind == "document.extract" and handler is None:
                from legal_workspace.documents.service import DocumentService

                def handler(_payload):
                    return DocumentService(self.database).process(claim)

            if handler is None:
                self.queue.fail(claim, "unsupported_job_kind", retryable=False)
                return True
            try:
                result = handler(claim.payload)
            except Exception as exc:
                logger.warning(
                    "job_handler_failed job_id=%s error_type=%s", claim.id, type(exc).__name__
                )
                self.queue.fail(claim, "handler_failed")
            else:
                if not self.queue.finish(claim, result):
                    logger.warning("job_result_discarded job_id=%s reason=lease_lost", claim.id)
        finally:
            with self._lock:
                self._current = None
        return True

    def run(self) -> None:
        now = time.time()
        with self.database.transaction() as connection:
            connection.execute(
                WorkerState.__table__.insert().values(
                    id=self.id,
                    started_at=now,
                    heartbeat_at=now,
                )
            )
        heartbeat = threading.Thread(target=self._heartbeat_loop, daemon=True)
        heartbeat.start()
        logger.info("worker_started worker_id=%s", self.id)
        try:
            while not self.stop_requested.is_set():
                if not self.run_once():
                    self.stop_requested.wait(self.settings.poll_interval_seconds)
        finally:
            self._heartbeat_stop.set()
            heartbeat.join()
            with self.database.transaction() as connection:
                connection.execute(
                    update(WorkerState)
                    .where(WorkerState.id == self.id)
                    .values(stopped_at=time.time())
                )
            logger.info("worker_stopped worker_id=%s", self.id)
