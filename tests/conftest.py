import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

from legal_workspace.config import Settings
from legal_workspace.persistence.database import Database


def pytest_addoption(parser):
    parser.addoption(
        "--ollama", action="store_true", help="Run real local Qwen/Ollama RAG integration test"
    )
    parser.addoption(
        "--real-model",
        action="store_true",
        help="Run offline ONNX smoke test against a prepared real model",
    )
    parser.addoption("--browser", action="store_true", help="Run the real Chromium acceptance test")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--browser"):
        for item in items:
            if "browser" in item.keywords:
                item.add_marker(pytest.mark.skip(reason="Enable with --browser; requires Chromium"))

    if not config.getoption("--real-model"):
        for item in items:
            if "real_model" in item.keywords:
                item.add_marker(
                    pytest.mark.skip(reason="Enable with --real-model after prepare-model")
                )

    if not config.getoption("--ollama"):
        for item in items:
            if "ollama" in item.keywords:
                item.add_marker(
                    pytest.mark.skip(reason="Enable with --ollama after ollama pull qwen3:4b")
                )


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "state", _env_file=None)


@pytest.fixture
def database(settings):
    db = Database(settings)
    db.initialize()
    yield db
    db.close()


def wait_until(callback, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = callback()
        if result:
            return result
        time.sleep(0.05)
    raise AssertionError("Timed out waiting for the expected application state")


class Runtime:
    def __init__(self, tmp_path: Path):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        self.root = tmp_path
        self.settings = Settings(
            data_dir=tmp_path / "state",
            port=self.port,
            poll_interval_seconds=0.05,
            heartbeat_interval_seconds=0.1,
            lease_seconds=1,
            worker_stale_after_seconds=0.5,
            _env_file=None,
        )
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.client = httpx.Client(base_url=self.base_url, timeout=2, trust_env=False)
        self.processes = []
        self.logs = []

    def start(self, command, *, code=None):
        environment = os.environ.copy()
        for field in (
            "data_dir",
            "port",
            "poll_interval_seconds",
            "heartbeat_interval_seconds",
            "lease_seconds",
            "worker_stale_after_seconds",
        ):
            environment[f"LEGAL_WORKSPACE_{field.upper()}"] = str(getattr(self.settings, field))
        environment["LEGAL_WORKSPACE_HOST"] = "127.0.0.1"
        log = (self.root / f"{command}-{len(self.processes)}.log").open("w+")
        arguments = (
            [sys.executable, "-c", code]
            if code
            else [
                sys.executable,
                "-m",
                "legal_workspace.main",
                command,
            ]
        )
        process = subprocess.Popen(
            arguments,
            cwd=self.root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        self.processes.append(process)
        self.logs.append(log)
        return process

    def start_web(self, *, code=None):
        process = self.start("serve", code=code)

        def healthy():
            if process.poll() is not None:
                log = self.logs[-1]
                log.flush()
                raise AssertionError(f"Web process exited: {Path(log.name).read_text()}")
            try:
                return self.client.get("/api/health").status_code == 200
            except httpx.TransportError:
                return False

        wait_until(healthy)
        return process

    @staticmethod
    def stop(process, *, kill=False):
        if process.poll() is None:
            process.kill() if kill else process.terminate()
            process.wait(timeout=10)

    def close(self):
        for process in reversed(self.processes):
            self.stop(process)
        self.client.close()
        for log in self.logs:
            log.close()


@pytest.fixture
def runtime(tmp_path):
    instance = Runtime(tmp_path)
    yield instance
    instance.close()
