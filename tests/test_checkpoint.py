from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from sae_pipeline.sae import checkpoint as checkpoint_module
from sae_pipeline.sae.checkpoint import (
    companion_state_path,
    load_sae_weights,
    normalize_hf_repo_id,
    resolve_checkpoint,
    save_checkpoint_pair,
)
from sae_pipeline.sae.train import build_sae


def _write_weights(path: Path, d_in: int = 3, d_sae: int = 5) -> None:
    model = build_sae("jumprelu", d_in=d_in, d_sae=d_sae)
    save_file(
        {key: value.detach().contiguous() for key, value in model.state_dict().items()},
        str(path),
    )


def test_local_latest_explicit_and_complete_pair_selection(tmp_path: Path):
    old = tmp_path / "sae_step_0000009.safetensors"
    newest = tmp_path / "sae_step_0000100.safetensors"
    _write_weights(old)
    _write_weights(newest)
    torch.save({"global_step": 9}, companion_state_path(old))

    assert resolve_checkpoint(tmp_path).weights_path == newest
    assert resolve_checkpoint(tmp_path, filename=old.name).weights_path == old
    resumed = resolve_checkpoint(tmp_path, require_trainer_state=True)
    assert resumed.weights_path == old
    assert resumed.trainer_state_path == companion_state_path(old)


def test_hf_repository_url_is_normalized_and_downloaded(monkeypatch, tmp_path: Path):
    downloaded = tmp_path / "sae_step_0000012.safetensors"
    _write_weights(downloaded)
    calls = []

    def fake_list_repo_files(*, repo_id, revision):
        calls.append(("list", repo_id, revision))
        return ["sae_step_0000002.safetensors", "sae_step_0000012.safetensors"]

    def fake_download(*, repo_id, filename, revision):
        calls.append(("download", repo_id, filename, revision))
        return str(downloaded)

    monkeypatch.setattr(checkpoint_module, "list_repo_files", fake_list_repo_files)
    monkeypatch.setattr(checkpoint_module, "hf_hub_download", fake_download)
    resolved = resolve_checkpoint(
        "https://huggingface.co/example/sae-repo/", revision="v2"
    )

    assert normalize_hf_repo_id("https://huggingface.co/example/sae-repo/") == "example/sae-repo"
    assert resolved.step == 12
    assert calls == [
        ("list", "example/sae-repo", "v2"),
        ("download", "example/sae-repo", "sae_step_0000012.safetensors", "v2"),
    ]


def test_legacy_weights_load_and_dimension_mismatch(tmp_path: Path):
    weights = tmp_path / "sae_step_0000001.safetensors"
    _write_weights(weights, d_in=3, d_sae=5)
    compatible = build_sae("jumprelu", d_in=3, d_sae=5)
    load_sae_weights(compatible, weights)

    incompatible = build_sae("jumprelu", d_in=3, d_sae=6)
    with pytest.raises(ValueError, match="incompatible tensor dimensions"):
        load_sae_weights(incompatible, weights)


def test_retention_only_removes_old_complete_pairs(tmp_path: Path):
    orphan = tmp_path / "sae_step_0000000.safetensors"
    _write_weights(orphan)
    model = build_sae("jumprelu", d_in=3, d_sae=5)

    for step in (1, 2, 3):
        save_checkpoint_pair(
            tmp_path,
            step,
            model.state_dict(),
            {"global_step": step},
            keep_last=2,
        )

    assert orphan.exists()
    assert not (tmp_path / "sae_step_0000001.safetensors").exists()
    assert not (tmp_path / "trainer_state_step_0000001.pt").exists()
    for step in (2, 3):
        assert (tmp_path / f"sae_step_{step:07d}.safetensors").exists()
        assert (tmp_path / f"trainer_state_step_{step:07d}.pt").exists()


def test_failed_save_does_not_commit_partial_pair(monkeypatch, tmp_path: Path):
    model = build_sae("jumprelu", d_in=3, d_sae=5)

    def fail_save(*args, **kwargs):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(checkpoint_module.torch, "save", fail_save)
    with pytest.raises(RuntimeError, match="simulated failure"):
        save_checkpoint_pair(
            tmp_path, 1, model.state_dict(), {"global_step": 1}, keep_last=2
        )
    assert list(tmp_path.iterdir()) == []
