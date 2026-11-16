from concurrent.futures import ThreadPoolExecutor
from threading import Event, get_ident

import pytest
from sqlalchemy import event

from legal_workspace.jobs.queue import JobQueue
from legal_workspace.persistence.database import Database


@pytest.fixture
def queue(database):
    clock = [1000.0]
    return JobQueue(database, clock=lambda: clock[0]), clock


def test_job_survives_database_reopen(database, settings):
    job_id = JobQueue(database).enqueue("system.check")
    database.close()
    reopened = Database(settings)
    try:
        reopened.initialize()
        queue = JobQueue(reopened)
        assert queue.get(job_id)["status"] == "queued"
        claim = queue.claim("worker", 30)
        assert queue.finish(claim, {"storage": "ok"})
        assert queue.get(job_id)["result"] == {"storage": "ok"}
    finally:
        reopened.close()


def test_priority_and_no_duplicate_claim(queue):
    jobs, _ = queue
    low = jobs.enqueue("low")
    high = jobs.enqueue("high", priority=10)
    assert jobs.claim("first", 30).id == high
    assert jobs.claim("second", 30).id == low
    assert jobs.claim("third", 30) is None


def test_competing_claims_only_deliver_job_once(queue):
    jobs, _ = queue
    job_id = jobs.enqueue("system.check")
    with ThreadPoolExecutor(max_workers=4) as executor:
        claims = list(executor.map(lambda worker: jobs.claim(str(worker), 30), range(4)))
    assert [claim.id for claim in claims if claim is not None] == [job_id]


def test_expired_job_is_recovered_and_old_owner_is_fenced(queue):
    jobs, clock = queue
    job_id = jobs.enqueue("system.check")
    old = jobs.claim("worker", 10)
    clock[0] += 11
    assert not jobs.finish(old, {"wrong": True})
    assert not jobs.renew(old, 10)
    new = jobs.claim("worker", 10)  # Even reusing the owner cannot reuse a previous lease token.
    assert new.id == old.id
    assert new.token != old.token
    assert not jobs.fail(old, "obsolete")
    assert not jobs.finish(old, {"wrong": True})
    assert jobs.finish(new, {"storage": "ok"})
    assert jobs.get(job_id)["attempts"] == 2


def test_heartbeat_extends_live_lease(queue):
    jobs, clock = queue
    jobs.enqueue("system.check")
    claim = jobs.claim("first", 10)
    clock[0] += 8
    assert jobs.renew(claim, 10)
    clock[0] += 4
    assert jobs.claim("second", 10) is None
    assert jobs.finish(claim, {})


def test_crash_retry_limit(queue):
    jobs, clock = queue
    job_id = jobs.enqueue("system.check", max_attempts=1)
    jobs.claim("crashed", 5)
    clock[0] += 6
    assert jobs.claim("replacement", 5) is None
    assert jobs.get(job_id)["status"] == "failed"
    assert jobs.get(job_id)["error_code"] == "lease_expired"


def test_handler_failure_backoff_and_limit(queue):
    jobs, clock = queue
    job_id = jobs.enqueue("system.check", max_attempts=2)
    first = jobs.claim("worker", 30)
    assert jobs.fail(first, "handler_failed")
    assert jobs.claim("worker", 30) is None
    clock[0] += 2
    second = jobs.claim("worker", 30)
    assert jobs.fail(second, "handler_failed")
    assert jobs.get(job_id)["status"] == "failed"
    clock[0] += 100
    assert jobs.claim("worker", 30) is None


def test_invalid_retry_limit(queue):
    with pytest.raises(ValueError):
        queue[0].enqueue("system.check", max_attempts=0)


def test_completion_rechecks_expiry_after_waiting_for_database_lock(queue, database):
    jobs, clock = queue
    jobs.enqueue("system.check")
    claim = jobs.claim("worker", 10)
    entering_write = Event()
    parent_thread = get_ident()

    def before_write(_connection, _cursor, statement, _parameters, _context, _executemany):
        if get_ident() != parent_thread and (
            statement == "BEGIN IMMEDIATE" or statement.startswith("UPDATE jobs")
        ):
            entering_write.set()

    event.listen(database.engine, "before_cursor_execute", before_write)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            with database.transaction(immediate=True):
                completing = executor.submit(jobs.finish, claim, {"storage": "ok"})
                assert entering_write.wait(timeout=2)
                clock[0] += 11
            assert not completing.result(timeout=2)
    finally:
        event.remove(database.engine, "before_cursor_execute", before_write)
