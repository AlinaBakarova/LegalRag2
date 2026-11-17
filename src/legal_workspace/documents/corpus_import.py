"""Import only a user-reviewed, local corpus; never fetch or invent legal text."""

import hashlib
import json
from datetime import date
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from legal_workspace.documents.limits import BASE_LIMITS
from legal_workspace.documents.parsers import DocumentError
from legal_workspace.documents.service import DocumentService


class Entry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    key: str = Field(min_length=1, max_length=200)
    file: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    title: str = Field(min_length=1)
    source_url: HttpUrl
    edition: str = Field(min_length=1)
    verified_on: date


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    documents: list[Entry]


def import_corpus(service: DocumentService, manifest_path: Path) -> list[str]:
    manifest = Manifest.model_validate(json.loads(manifest_path.read_text(encoding="utf-8")))
    root = manifest_path.resolve().parent
    prepared = []
    keys = set()
    for entry in manifest.documents:
        path = (root / entry.file).resolve()
        if not path.is_relative_to(root) or entry.key in keys:
            raise DocumentError("invalid_manifest_path_or_duplicate")
        keys.add(entry.key)
        if path.stat().st_size > BASE_LIMITS.file_bytes:
            raise DocumentError("file_size_limit")
        with path.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        if checksum != entry.sha256:
            raise DocumentError("corpus_checksum_mismatch")
        prepared.append((entry, path))
    ids = []
    for entry, path in prepared:
        with path.open("rb") as stream:
            ids.append(
                service.upload_stream(
                    path.name,
                    stream,
                    corpus_key=entry.key,
                    expected_sha=entry.sha256,
                    source=entry.model_dump(mode="json", exclude={"file", "sha256", "key"}),
                )
            )
    return ids
