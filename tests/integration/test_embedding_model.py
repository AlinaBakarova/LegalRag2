import pytest

from legal_workspace.retrieval.embeddings import (
    FILES,
    REPOSITORY,
    SearchError,
    model_path,
    prepare_model,
    verified_manifest,
)


def test_model_install_is_commit_pinned_and_verified(tmp_path, monkeypatch):
    import huggingface_hub

    calls = []

    class Info:
        sha = "a" * 40

    class Api:
        def model_info(self, repository):
            assert repository == REPOSITORY
            return Info()

    def download(repository, revision, local_dir, allow_patterns, max_workers):
        assert repository == REPOSITORY
        assert revision == Info.sha
        local_dir.mkdir()
        for name in allow_patterns:
            (local_dir / name).write_bytes(b"SYNTHETIC MODEL ARTIFACT TEST ONLY")
        calls.append(revision)

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    manifest = prepare_model(tmp_path)
    assert manifest["revision"] == Info.sha
    assert prepare_model(tmp_path) == manifest
    assert calls == [Info.sha]
    path = model_path(tmp_path)
    (path / FILES[0]).write_bytes(b"corrupted")
    with pytest.raises(SearchError, match="model_not_ready"):
        verified_manifest(path)


@pytest.mark.real_model
def test_real_local_model_russian_embeddings():
    import numpy as np

    from legal_workspace.config import Settings
    from legal_workspace.retrieval.embeddings import get_embedder

    model = get_embedder(Settings().data_dir)
    vectors = model.embed(
        [
            "Как перечислить деньги за выполненную работу?",
            "Оплата услуг осуществляется банковским переводом.",
            "Оборудование хранится на складе при определённой температуре.",
        ]
    )
    assert vectors.shape == (3, 384)
    assert np.isfinite(vectors).all()
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
    assert vectors[0] @ vectors[1] > vectors[0] @ vectors[2]
