"""Validated local runtime settings; never log the settings object or environment."""

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LEGAL_WORKSPACE_", env_file=".env", extra="ignore"
    )

    data_dir: Path = Field(default_factory=lambda: Path.home() / ".local/share/legal-workspace")
    host: Literal["127.0.0.1", "localhost", "::1"] = "127.0.0.1"
    port: int = Field(default=8080, ge=1024, le=65535)
    poll_interval_seconds: float = Field(default=0.5, gt=0, le=30)
    heartbeat_interval_seconds: float = Field(default=2, gt=0, le=30)
    lease_seconds: float = Field(default=30, gt=0, le=3600)
    worker_stale_after_seconds: float = Field(default=10, gt=0, le=3600)

    @model_validator(mode="after")
    def validate_timing(self) -> "Settings":
        if self.lease_seconds < 3 * self.heartbeat_interval_seconds:
            raise ValueError("lease_seconds must be at least three heartbeat intervals")
        if self.worker_stale_after_seconds < 3 * self.heartbeat_interval_seconds:
            raise ValueError(
                "worker_stale_after_seconds must be at least three heartbeat intervals"
            )
        self.data_dir = self.data_dir.expanduser().resolve()
        return self

    @property
    def database_path(self) -> Path:
        return self.data_dir / "workspace.sqlite3"
