import os
import subprocess
import sys

import pytest
from pydantic import ValidationError

from legal_workspace.config import Settings


def test_environment_settings_and_absolute_data_path(tmp_path, monkeypatch):
    monkeypatch.setenv("LEGAL_WORKSPACE_DATA_DIR", str(tmp_path / "persistent"))
    monkeypatch.setenv("LEGAL_WORKSPACE_PORT", "8123")
    config = Settings(_env_file=None)
    assert config.database_path == tmp_path / "persistent/workspace.sqlite3"
    assert config.port == 8123


@pytest.mark.parametrize(
    "values",
    [
        {"host": "0.0.0.0"},
        {"port": 70000},
        {"poll_interval_seconds": 0},
        {"lease_seconds": 1},
        {"worker_stale_after_seconds": 1},
    ],
)
def test_invalid_settings_rejected(values):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **values)


def test_configuration_errors_do_not_print_input(tmp_path):
    marker = "NON_REAL_SECRET_DO_NOT_LOG"
    environment = dict(os.environ, LEGAL_WORKSPACE_PORT=marker)
    result = subprocess.run(
        [sys.executable, "-m", "legal_workspace.main", "init"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "port" in result.stderr
    assert marker not in result.stdout + result.stderr
