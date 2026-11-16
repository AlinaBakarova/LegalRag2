import logging

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from legal_workspace.api import create_api


@pytest.fixture
def client(settings):
    with TestClient(create_api(settings), base_url="http://127.0.0.1") as instance:
        yield instance


def test_health_and_offline_worker_are_distinct(client):
    assert client.get("/api/health").json() == {"status": "ok", "database": "ok"}
    response = client.get("/api/status")
    assert response.status_code == 200
    assert response.json()["worker"] == {"online": False, "active_count": 0}
    assert response.json()["workspace"]["name"] == "Legal AI Workspace"


def test_schedule_and_read_job(client):
    response = client.post("/api/jobs/check")
    assert response.status_code == 202
    job_id = response.json()["id"]
    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "queued"
    assert "payload" not in job
    assert "lease_token" not in job
    assert client.get("/api/jobs").json() == [job]
    assert client.get("/api/jobs/missing").status_code == 404


def test_rejects_foreign_host_and_cross_origin_mutation(client):
    assert client.get("/api/health", headers={"host": "untrusted.example"}).status_code == 400
    assert (
        client.post("/api/jobs/check", headers={"origin": "https://untrusted.example"}).status_code
        == 403
    )
    assert client.post("/api/jobs/check", headers={"origin": "http://127.0.0.1"}).status_code == 202


def test_storage_errors_do_not_expose_parameters(client, monkeypatch, caplog):
    marker = "NON_REAL_SECRET_DO_NOT_LOG"

    def broken():
        raise OperationalError("select private", {"key": marker}, Exception(marker))

    monkeypatch.setattr(client.app.state.service, "status", broken)
    with caplog.at_level(logging.WARNING):
        response = client.get("/api/status")
    assert response.status_code == 503
    assert marker not in response.text + caplog.text
