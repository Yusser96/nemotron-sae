from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from sae_pipeline.cache.manifest import CacheManifest
from sae_pipeline.cache.writer import ShardWriter
from sae_pipeline.config import SAECfg
from sae_pipeline.sae.train import build_sae, train_sae


def _make_cache(path: Path) -> None:
    generator = torch.Generator().manual_seed(123)
    activations = torch.randn(53, 4, generator=generator)
    manifest = CacheManifest(
        run_id="resume",
        model="synthetic",
        dtype="float32",
        layer=0,
        component="resid_post",
        d_activation=4,
    )
    with ShardWriter(
        out_dir=path,
        d_activation=4,
        shard_size_bytes=80,
        dtype=torch.float32,
        shuffle_seed=3,
        manifest=manifest,
    ) as writer:
        writer.add(activations)


def test_interrupted_resume_matches_uninterrupted_cpu_run(tmp_path: Path, monkeypatch):
    # Plotting is unrelated to training continuity and dominates this tiny test.
    monkeypatch.setattr("sae_pipeline.sae.train.plot_training_curves", lambda **_: None)
    cache = tmp_path / "cache"
    _make_cache(cache)
    common = dict(
        d_sae=7,
        l0_target=2,
        lr=1e-3,
        batch_size=5,
        n_steps=6,
        warmup_steps=2,
        l0_warmup_steps=3,
        ckpt_every=2,
        log_every=1,
        n_batches_in_buffer=3,
        keep_last_checkpoints=3,
    )

    torch.manual_seed(777)
    full = train_sae(
        SAECfg(**common), cache, 4, "jumprelu", 7, 2, tmp_path / "full", "cpu"
    )

    # A different process seed demonstrates that resume restores checkpoint RNG.
    torch.manual_seed(9999)
    resumed = train_sae(
        SAECfg(
            **common,
            checkpoint_source=tmp_path / "full",
            checkpoint_mode="resume",
            checkpoint_filename="sae_step_0000004.safetensors",
        ),
        cache,
        4,
        "jumprelu",
        7,
        2,
        tmp_path / "resumed",
        "cpu",
    )

    full_state = load_file(str(full))
    resumed_state = load_file(str(resumed))
    assert full_state.keys() == resumed_state.keys()
    assert all(torch.equal(full_state[key], resumed_state[key]) for key in full_state)


def test_weight_only_finetune_starts_at_one_and_preserves_source(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr("sae_pipeline.sae.train.plot_training_curves", lambda **_: None)
    cache = tmp_path / "cache"
    _make_cache(cache)
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source = source_dir / "sae_step_0000099.safetensors"
    source_model = build_sae("jumprelu", d_in=4, d_sae=7)
    save_file(
        {
            key: value.detach().cpu().contiguous()
            for key, value in source_model.state_dict().items()
        },
        str(source),
    )
    original_source = source.read_bytes()

    result = train_sae(
        SAECfg(
            d_sae=7,
            l0_target=2,
            batch_size=5,
            n_steps=1,
            warmup_steps=1,
            l0_warmup_steps=1,
            ckpt_every=1,
            log_every=1,
            n_batches_in_buffer=2,
            checkpoint_source=source,
        ),
        cache,
        4,
        "jumprelu",
        7,
        2,
        tmp_path / "finetuned",
        "cpu",
    )

    assert result.name == "sae_step_0000001.safetensors"
    assert source.read_bytes() == original_source
    assert (tmp_path / "finetuned" / "trainer_state_step_0000001.pt").exists()
