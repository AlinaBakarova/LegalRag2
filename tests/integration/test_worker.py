import logging

from legal_workspace.jobs.queue import JobQueue
from legal_workspace.jobs.worker import Worker


def test_worker_executes_storage_check(database, settings):
    queue = JobQueue(database)
    job_id = queue.enqueue("system.check")
    assert Worker(database, settings).run_once()
    assert queue.get(job_id)["status"] == "succeeded"
    assert queue.get(job_id)["result"] == {"storage": "ok"}


def test_unknown_handler_fails_without_retries(database, settings):
    queue = JobQueue(database)
    job_id = queue.enqueue("unsupported")
    Worker(database, settings).run_once()
    assert queue.get(job_id)["status"] == "failed"
    assert queue.get(job_id)["error_code"] == "unsupported_job_kind"


def test_handler_exception_does_not_expose_payload(database, settings, caplog):
    marker = "NON_REAL_SECRET_DO_NOT_LOG"

    def broken(payload):
        raise RuntimeError(payload["private"])

    queue = JobQueue(database)
    job_id = queue.enqueue("broken", {"private": marker})
    with caplog.at_level(logging.WARNING):
        Worker(database, settings, handlers={"broken": broken}).run_once()
    assert marker not in caplog.text
    assert "RuntimeError" in caplog.text
    assert queue.get(job_id)["error_code"] == "handler_failed"
    assert queue.get(job_id)["status"] == "queued"
