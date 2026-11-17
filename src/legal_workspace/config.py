"""Validated local runtime settings; never log the settings object or environment."""

from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LEGAL_WORKSPACE_", env_file=".env", extra="ignore"
    )

    data_dir: Path = Field(default_factory=lambda: Path.home() / ".local/share/legal-workspace")
    host: Literal["127.0.0.1", "localhost", "::1"] = "127.0.0.1"
    embedding_threads: int = Field(default=2, ge=1, le=8)
    ollama_url: str = "http://127.0.0.1:11434"
    llm_model: str = Field(default="qwen3:4b", min_length=1, max_length=120)
    llm_timeout_seconds: float = Field(default=180, gt=0, le=600)
    llm_total_timeout_seconds: float = Field(default=360, gt=0, le=1800)
    llm_context_tokens: int = Field(default=8192, ge=4096, le=32768)
    llm_output_tokens: int = Field(default=1024, ge=256, le=2048)
    llm_max_calls: int = Field(default=3, ge=2, le=4)
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
        parsed = urlsplit(self.ollama_url)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
            or parsed.username
            or parsed.password
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("ollama_url must be a local HTTP origin")
        if parsed.port is not None and not 1 <= parsed.port <= 65535:
            raise ValueError("Invalid local Ollama port")
        if "cloud" in self.llm_model.lower() or any(c.isspace() for c in self.llm_model):
            raise ValueError("Only local Ollama model tags are supported")
        self.ollama_url = self.ollama_url.rstrip("/")
        self.data_dir = self.data_dir.expanduser().resolve()
        return self

    @property
    def database_path(self) -> Path:
        return self.data_dir / "workspace.sqlite3"
