import textwrap
import time

from conftest import wait_until

from legal_workspace.jobs.queue import JobQueue
from legal_workspace.persistence.database import Database


def test_web_and_worker_restart_preserve_state(runtime):
    web = runtime.start_web()
    workspace = runtime.client.get("/api/status").json()["workspace"]
    job_id = runtime.client.post("/api/jobs/check").json()["id"]
    assert runtime.client.get(f"/api/jobs/{job_id}").json()["status"] == "queued"
    runtime.stop(web)
    runtime.start_web()
    assert runtime.client.get("/api/status").json()["workspace"] == workspace
    assert runtime.client.get(f"/api/jobs/{job_id}").json()["status"] == "queued"

    worker = runtime.start("worker")
    wait_until(lambda: runtime.client.get(f"/api/jobs/{job_id}").json()["status"] == "succeeded")
    assert runtime.client.get("/api/status").json()["worker"]["online"]
    runtime.stop(worker)
    wait_until(lambda: not runtime.client.get("/api/status").json()["worker"]["online"])
    next_id = runtime.client.post("/api/jobs/check").json()["id"]
    runtime.start("worker")
    wait_until(lambda: runtime.client.get(f"/api/jobs/{next_id}").json()["status"] == "succeeded")
    result = runtime.client.get(f"/api/jobs/{job_id}").json()
    assert result["attempts"] == 1
    assert result["result"] == {"storage": "ok"}


def test_killed_worker_lease_is_recovered_by_new_process(runtime):
    runtime.start_web()
    job_id = runtime.client.post("/api/jobs/check").json()["id"]
    # Exercise the real worker heartbeat with a controlled long-running test handler.
    slow_worker = runtime.start(
        "slow-worker",
        code=textwrap.dedent("""
        import time
        from legal_workspace.config import Settings
        from legal_workspace.persistence.database import Database
        from legal_workspace.jobs.worker import Worker
        settings = Settings()
        database = Database(settings)
        database.initialize()
        def slow_check(payload):
            time.sleep(60)
            return {"storage": "ok"}
        Worker(database, settings, handlers={"system.check": slow_check}).run()
    """),
    )
    wait_until(lambda: runtime.client.get(f"/api/jobs/{job_id}").json()["status"] == "running")
    # The lease remains live beyond its original expiry while the handler is running.
    db = Database(runtime.settings)
    try:
        queue = JobQueue(db)
        first_expiry = queue.get(job_id)["lease_expires_at"]
        wait_until(lambda: time.time() > first_expiry + 0.1)
        assert queue.get(job_id)["lease_expires_at"] > time.time()
        assert queue.claim("competing-worker", 1) is None
        runtime.stop(slow_worker, kill=True)
        wait_until(lambda: not runtime.client.get("/api/status").json()["worker"]["online"])
        runtime.start("worker")
        wait_until(
            lambda: runtime.client.get(f"/api/jobs/{job_id}").json()["status"] == "succeeded"
        )
        assert queue.get(job_id)["attempts"] == 2
    finally:
        db.close()
