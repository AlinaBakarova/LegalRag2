"""Compact local RAG diagnostics. Only model responses contain source-derived text."""

import json
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class RagTrace:
    message_id: str
    job_id: str
    question_id: str | None = None
    stage: str = "history"

    def emit(self, event: str, **fields) -> None:
        # JSON escapes newlines/control characters in model text: one event is one log line.
        logger.info(
            json.dumps(
                {
                    "time": datetime.now(UTC).isoformat(timespec="milliseconds"),
                    "event": event,
                    "message_id": self.message_id,
                    "question_id": self.question_id,
                    "job_id": self.job_id,
                    "stage": self.stage,
                    **fields,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )


class _PrivateRotatingHandler(RotatingFileHandler):
    def _open(self):
        descriptor = os.open(self.baseFilename, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        return os.fdopen(descriptor, "a", encoding="utf-8")


@contextmanager
def rag_log_file(data_dir: Path):
    """CLI worker keeps JSONL diagnostics across restarts, with bounded disk usage."""
    directory = data_dir / "logs"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    handler = _PrivateRotatingHandler(
        directory / "rag.jsonl", maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    previous_level = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        yield
    finally:
        logger.removeHandler(handler)
        handler.close()
        logger.setLevel(previous_level)
