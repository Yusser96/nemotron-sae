"""Streaming SAE steps: parity with cached training, log rows and resumed penalty state."""

import json
import math

import pytest
import torch
from safetensors.torch import load_file, save_file

from sae_pipeline.config import EarlyStoppingCfg, SAECfg
from sae_pipeline.sae.checkpoint_source import resolve_checkpoint
from sae_pipeline.sae.train import train_sae
from sae_pipeline.sae.train_streaming import streaming_step, write_checkpoint
from tests.sae.streaming_fakes import (
    BATCH,
    D_IN,
    D_SAE,
    L0_TARGET,
    make_sae_cfg,
    make_site,
    write_warm_start,
)
from tests.synthetic import write_cache


def test_streaming_step_matches_one_cached_training_step(tmp_path, monkeypatch):
    monkeypatch.setattr("sae_pipeline.sae.train.plot_training_curves", lambda **_: None)
    x = torch.randn(BATCH, D_IN, generator=torch.Generator().manual_seed(0)) * 2.0
    cfg = make_sae_cfg(write_warm_start(tmp_path / "warm.safetensors"))

    cached = train_sae(
        cfg=cfg,
        cache_dir=write_cache(tmp_path / "cache", x),
        d_in=D_IN,
        arch="jumprelu",
        d_sae=D_SAE,
        l0_target=L0_TARGET,
        out_dir=tmp_path / "cached",
        device="cpu",
    )
    cached_row = json.loads((tmp_path / "cached" / "train_log.jsonl").read_text().splitlines()[0])

    site = make_site(tmp_path, cfg)
    entry = streaming_step(site, x, step=1)
    site.close()

    assert entry is not None
    for key in ("loss", "reconstruction_loss", "l0_penalty", "inverse_simpson_penalty", "lr", "hard_l0"):
        assert getattr(entry, key) == pytest.approx(cached_row[key], rel=1e-5, abs=1e-7), key
    cached_weights = load_file(str(cached))
    for key, value in site.sae.state_dict().items():
        torch.testing.assert_close(value, cached_weights[key], rtol=1e-5, atol=1e-6)


def test_streaming_step_logs_only_every_log_every_steps(tmp_path):
    site = make_site(tmp_path, make_sae_cfg(write_warm_start(tmp_path / "warm.safetensors"), log_every=2))
    x = torch.randn(BATCH, D_IN)

    assert streaming_step(site, x, step=1) is None
    assert streaming_step(site, x, step=2) is not None
    site.close()


def test_streaming_step_applies_the_early_stop_lr_scale(tmp_path):
    site = make_site(tmp_path, make_sae_cfg(write_warm_start(tmp_path / "warm.safetensors"), warmup_steps=0))
    site.early_stop.lr_scale = 0.25

    entry = streaming_step(site, torch.randn(BATCH, D_IN), step=1)
    site.close()

    assert entry.lr == pytest.approx(0.25e-2)


@pytest.mark.parametrize(
    "overrides",
    [
        {"feature_use_strategy": "frequency"},
        {"activation_centering": "mean"},
        {"active_subspace_rank": 2},
        {"lr_schedule": "cosine_decay"},
    ],
)
def test_options_that_need_a_cache_or_a_horizon_are_rejected(tmp_path, overrides):
    with pytest.raises((NotImplementedError, ValueError)):
        make_site(tmp_path, make_sae_cfg(write_warm_start(tmp_path / "warm.safetensors"), **overrides))


def test_nonfinite_warm_start_is_rejected_before_training(tmp_path):
    path = write_warm_start(tmp_path / "warm.safetensors")
    tensors = load_file(str(path))
    tensors["W_enc"][0, 0] = math.nan
    save_file(tensors, str(path))
    with pytest.raises(ValueError, match="non-finite checkpoint parameter W_enc"):
        make_site(tmp_path, make_sae_cfg(path))


def test_a_resumed_site_continues_the_inverse_simpson_window(tmp_path):
    cfg = make_sae_cfg(write_warm_start(tmp_path / "warm.safetensors"), inverse_simpson_window_steps=8)
    site = make_site(tmp_path, cfg)
    for step in (1, 2):
        streaming_step(site, torch.randn(BATCH, D_IN, generator=torch.Generator().manual_seed(step)), step)
    write_checkpoint(site, 2)
    site.close()

    resumed = make_site(tmp_path, cfg, resume_checkpoint=resolve_checkpoint(site.out_dir, require_trainer_state=True))
    resumed.close()

    assert site.inverse_simpson_usage.sum() > 0
    torch.testing.assert_close(resumed.inverse_simpson_usage, site.inverse_simpson_usage)


def test_a_resumed_site_continues_the_auxk_dead_counter(tmp_path):
    cfg = make_sae_cfg(write_warm_start(tmp_path / "warm.safetensors"), auxk_scale=1 / 32, auxk_k=2, auxk_dead_tokens=BATCH)
    site = make_site(tmp_path, cfg)
    with torch.no_grad():
        site.sae.log_theta[:4] = 10.0  # four features that never fire
    for step in (1, 2):
        streaming_step(site, torch.randn(BATCH, D_IN, generator=torch.Generator().manual_seed(step)), step)
    write_checkpoint(site, 2)
    site.close()

    resumed = make_site(tmp_path, cfg, resume_checkpoint=resolve_checkpoint(site.out_dir, require_trainer_state=True))
    resumed.close()

    assert bool((site.tokens_since_fired[:4] == 2 * BATCH).all())
    torch.testing.assert_close(resumed.tokens_since_fired, site.tokens_since_fired)


def test_streaming_resets_resample_unused_features_and_survive_a_resume(tmp_path):
    cfg = make_sae_cfg(
        write_warm_start(tmp_path / "warm.safetensors"),
        feature_use_strategy="residual_reset",
        residual_reset_start_step=2,
        residual_reset_every_steps=10,
        residual_reset_max_features=4,
        residual_reset_pool_size=BATCH,
        auxk_scale=1 / 32,
    )
    site = make_site(tmp_path, cfg)
    with torch.no_grad():
        site.sae.log_theta[:4] = 10.0  # four features that never fire
    for step in (1, 2):
        streaming_step(site, torch.randn(BATCH, D_IN, generator=torch.Generator().manual_seed(step)), step)
    write_checkpoint(site, 2)
    site.close()

    resumed = make_site(tmp_path, cfg, resume_checkpoint=resolve_checkpoint(site.out_dir, require_trainer_state=True))
    resumed.close()

    assert site.feature_resets == 4
    assert bool((site.sae.log_theta[:4] < 10.0).all())  # recalibrated to fire again
    assert bool((site.tokens_since_fired[:4] == 0).all())  # a replaced latent starts a new history
    assert (resumed.feature_resets, resumed.last_reset_step) == (4, 2)


def test_the_log_row_tells_why_auxk_dead_latents_are_inactive(tmp_path):
    cfg = make_sae_cfg(write_warm_start(tmp_path / "warm.safetensors"), auxk_scale=1 / 32, auxk_k=2, auxk_dead_tokens=BATCH)
    site = make_site(tmp_path, cfg)
    with torch.no_grad():
        site.sae.log_theta[:4] = 10.0  # four features that never fire
    rows = [
        streaming_step(site, torch.randn(BATCH, D_IN, generator=torch.Generator().manual_seed(step)), step)
        for step in (1, 2)
    ]
    site.close()

    assert rows[1].dead_latents >= 4
    assert rows[1].dead_below_threshold + rows[1].dead_nonpositive == rows[1].dead_latents
    assert rows[1].auxk_ratio > 0


def test_streaming_seed_and_target_schedule_match_cached_recipe(tmp_path):
    cfg = make_sae_cfg(write_warm_start(tmp_path / "warm.safetensors"), seed=123,
                   l0_target_start=9, l0_target_warmup_steps=4)
    site = make_site(tmp_path, cfg)
    first_random = torch.rand(5)
    x = torch.randn(BATCH, D_IN, generator=torch.Generator().manual_seed(0))
    entry = streaming_step(site, x, 1)
    site.close()
    torch.manual_seed(999)
    again = make_site(tmp_path, cfg)
    torch.testing.assert_close(torch.rand(5), first_random)
    again.close()
    assert entry.target_l0 == 7.5


def test_cached_training_rejects_unimplemented_objective_stop_before_io(tmp_path):
    with pytest.raises(ValueError, match="stability"):
        train_sae(cfg=SAECfg(), cache_dir=tmp_path / "missing", d_in=D_IN,
                  arch="jumprelu", d_sae=D_SAE, l0_target=L0_TARGET, out_dir=tmp_path / "out",
                  early_stopping=EarlyStoppingCfg(enabled=True, mode="objective", patience=4))
    assert not (tmp_path / "out").exists()
