"""Only prepare_model uses the network. Runtime inference is CPU-only and offline."""

import hashlib
import json
import re
import shutil
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
from threading import Lock
from uuid import uuid4

import numpy as np
from filelock import FileLock

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
REPOSITORY = "qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q"
FILES = (
    "model_optimized.onnx",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "config.json",
)


class SearchError(ValueError):
    """Safe diagnostic code; no query/document text."""


def model_path(data_dir: Path) -> Path:
    return data_dir / "models" / "multilingual-minilm"


def verified_manifest(path: Path) -> dict:
    try:
        manifest = json.loads((path / "manifest.json").read_text())
        if manifest["model"] != MODEL or manifest["repository"] != REPOSITORY:
            raise ValueError
        if not re.fullmatch(r"[0-9a-f]{40}", manifest["revision"]):
            raise ValueError
        if set(manifest["files"]) != set(FILES):
            raise ValueError
        for name, expected in manifest["files"].items():
            with (path / name).open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                    raise ValueError
        return manifest
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise SearchError("model_not_ready") from exc


def prepare_model(data_dir: Path) -> dict:
    from huggingface_hub import HfApi, snapshot_download

    path = model_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path.parent / ".prepare.lock"), timeout=600):
        if path.exists():
            return verified_manifest(path)
        # Resolve main once, then download a fixed commit; no remote Python is executed.
        revision = HfApi().model_info(REPOSITORY).sha
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise SearchError("model_metadata_invalid")
        temporary = path.parent / (".download-" + uuid4().hex)
        try:
            snapshot_download(
                REPOSITORY,
                revision=revision,
                local_dir=temporary,
                allow_patterns=list(FILES),
                max_workers=2,
            )
            checksums = {}
            for name in FILES:
                with (temporary / name).open("rb") as stream:
                    checksums[name] = hashlib.file_digest(stream, "sha256").hexdigest()
            manifest = {
                "model": MODEL,
                "repository": REPOSITORY,
                "revision": revision,
                "files": checksums,
            }
            (temporary / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))
            verified_manifest(temporary)
            temporary.rename(path)
            return manifest
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)


class LocalEmbedder:
    dimension = 384
    max_tokens = 512

    def __init__(self, data_dir: Path, threads: int = 2):
        path = model_path(data_dir)
        manifest = verified_manifest(path)
        from fastembed import TextEmbedding
        from tokenizers import Tokenizer

        self.identity = hashlib.sha256(
            json.dumps(
                {"manifest": manifest, "backend": version("fastembed")}, sort_keys=True
            ).encode()
        ).hexdigest()
        self.tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        self.model = TextEmbedding(
            model_name=MODEL,
            specific_model_path=str(path),
            local_files_only=True,
            threads=threads,
            providers=["CPUExecutionProvider"],
        )
        self.lock = Lock()

    def token_count(self, text: str) -> int:
        return len(self.tokenizer.encode(text).ids)

    def embed(self, texts: list[str]) -> np.ndarray:
        if any(self.token_count(text) > self.max_tokens for text in texts):
            raise SearchError("embedding_context_exceeded")
        with self.lock:
            vectors = np.asarray(list(self.model.embed(texts, batch_size=16)), dtype="<f4")
        if vectors.shape != (len(texts), self.dimension) or not np.isfinite(vectors).all():
            raise SearchError("invalid_embeddings")
        norm = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norm == 0):
            raise SearchError("invalid_embeddings")
        return vectors / norm


@lru_cache(maxsize=2)
def get_embedder(data_dir: Path, threads: int = 2) -> LocalEmbedder:
    return LocalEmbedder(data_dir, threads)
