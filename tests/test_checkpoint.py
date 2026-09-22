from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from sae_pipeline.sae import checkpoint as checkpoint_module
from sae_pipeline.sae.checkpoint import (
    capture_random_state,
    companion_state_path,
    load_sae_weights,
    load_trainer_state,
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


def test_sae_lens_jumprelu_weights_are_adapted(tmp_path: Path):
    model = build_sae("jumprelu", d_in=3, d_sae=5)
    state = {key: value.detach().contiguous() for key, value in model.state_dict().items()}
    state["W_dec"] = state["W_dec"].T.contiguous()
    state["threshold"] = state.pop("log_theta").exp()
    weights = tmp_path / "sae_weights.safetensors"
    save_file(state, str(weights))

    loaded = build_sae("jumprelu", d_in=3, d_sae=5)
    load_sae_weights(loaded, weights)
    assert torch.equal(loaded.W_dec, state["W_dec"].T)
    assert torch.equal(loaded.theta, state["threshold"])


def test_hf_explicit_arbitrary_filename_is_allowed_for_finetuning(monkeypatch, tmp_path: Path):
    downloaded = tmp_path / "sae_weights.safetensors"
    _write_weights(downloaded)
    calls = []

    def fake_download(*, repo_id, filename, revision):
        calls.append((repo_id, filename, revision))
        return str(downloaded)

    monkeypatch.setattr(checkpoint_module, "hf_hub_download", fake_download)
    resolved = resolve_checkpoint(
        "example/sae-repo",
        filename="L2_resid_post/w16384_l0_10/sae_weights.safetensors",
    )

    assert resolved.step == -1
    assert calls == [
        ("example/sae-repo", "L2_resid_post/w16384_l0_10/sae_weights.safetensors", "main")
    ]


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


def test_arbitrary_named_local_file_is_usable_without_resume(tmp_path: Path):
    weights = tmp_path / "best_model.safetensors"
    _write_weights(weights)

    resolved = resolve_checkpoint(weights)
    assert resolved.weights_path == weights
    assert resolved.trainer_state_path is None


def test_arbitrary_named_local_file_rejected_for_resume(tmp_path: Path):
    weights = tmp_path / "best_model.safetensors"
    _write_weights(weights)

    with pytest.raises(ValueError, match="Resume requires"):
        resolve_checkpoint(weights, require_trainer_state=True)


def test_local_directory_scan_is_not_recursive(tmp_path: Path):
    top = tmp_path / "sae_step_0000001.safetensors"
    _write_weights(top)
    nested_dir = tmp_path / "old_subrun"
    nested_dir.mkdir()
    nested = nested_dir / "sae_step_0000999.safetensors"
    _write_weights(nested)

    assert resolve_checkpoint(tmp_path).weights_path == top


def test_hf_explicit_filename_resume_requires_companion_state(monkeypatch, tmp_path: Path):
    def fake_list_repo_files(*, repo_id, revision):
        return ["sae_step_0000012.safetensors"]

    def fake_download(*, repo_id, filename, revision):
        raise AssertionError("should not download when companion state is missing")

    monkeypatch.setattr(checkpoint_module, "list_repo_files", fake_list_repo_files)
    monkeypatch.setattr(checkpoint_module, "hf_hub_download", fake_download)

    with pytest.raises(FileNotFoundError, match="Resume requires companion state"):
        resolve_checkpoint(
            "example/sae-repo",
            filename="sae_step_0000012.safetensors",
            require_trainer_state=True,
        )


def test_trainer_state_round_trips_with_weights_only(tmp_path: Path):
    path = tmp_path / "trainer_state.pt"
    state = {
        "global_step": 12,
        "random_state": capture_random_state(),
        "optimizer": {"state": {}, "param_groups": []},
    }
    torch.save(state, path)

    loaded = load_trainer_state(path)

    assert loaded["global_step"] == 12
    assert loaded["random_state"]["numpy"][0] == state["random_state"]["numpy"][0]


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
