"""Streaming SAE training: per-step parity with cached training, evaluation, loop."""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from sae_pipeline.cache.manifest import CacheManifest
from sae_pipeline.cache.reader import ShardReader
from sae_pipeline.cache.writer import ShardWriter
from sae_pipeline.config import EarlyStoppingCfg, SAECfg
from sae_pipeline.sae.jumprelu import JumpReLUSAE
from sae_pipeline.sae.objective import coefficients_at, training_loss
from sae_pipeline.sae.train import train_sae
from sae_pipeline.sae.train_streaming import (
    evaluate_and_update,
    evaluate_site,
    init_streaming_site,
    record_baseline,
    streaming_step,
)

D_IN, D_SAE, L0_TARGET, BATCH = 6, 16, 3, 32


def _write_cache(path: Path, x: torch.Tensor, *, language: str | None = None) -> Path:
    manifest = CacheManifest(
        run_id="streaming-test",
        model="synthetic",
        dtype="float32",
        layer=0,
        component="resid_post",
        d_activation=x.shape[1],
        partition="validation" if language else "train",
        language=language,
    )
    with ShardWriter(
        out_dir=path, d_activation=x.shape[1], shard_size_bytes=4096, dtype=torch.float32, manifest=manifest
    ) as writer:
        writer.add(x)
    writer.manifest.complete = True
    writer.manifest.write(path / "manifest.json")
    return path


def _warm_start(path: Path) -> Path:
    torch.manual_seed(3)
    sae = JumpReLUSAE(d_in=D_IN, d_sae=D_SAE, bandwidth=0.05)
    save_file({key: value.detach().contiguous() for key, value in sae.state_dict().items()}, str(path))
    return path


def _sae_cfg(checkpoint: Path, **overrides) -> SAECfg:
    values = dict(
        arch="jumprelu",
        d_sae=D_SAE,
        l0_target=L0_TARGET,
        lr=1.0e-2,
        batch_size=BATCH,
        n_steps=1,
        warmup_steps=2,
        l0_warmup_steps=2,
        bandwidth=0.05,
        lr_schedule="cosine_warmup_constant",
        reconstruction_loss="vector_sum",
        gradient_clip_norm=1.0,
        inverse_simpson_floor=1.0,
        inverse_simpson_max_scale=5.0,
        log_every=1,
        ckpt_every=1,
        n_batches_in_buffer=1,
        checkpoint_source=checkpoint,
        checkpoint_mode="finetune",
    )
    values.update(overrides)
    return SAECfg(**values)


def _site(tmp_path: Path, cfg: SAECfg, validation_dirs=None, probe_tokens=BATCH, **kwargs):
    return init_streaming_site(
        slug="site",
        sae_cfg=cfg,
        d_in=D_IN,
        arch="jumprelu",
        d_sae=D_SAE,
        l0_target=L0_TARGET,
        out_dir=tmp_path / "streaming",
        device="cpu",
        validation_cache_dirs=validation_dirs or {},
        probe_tokens=probe_tokens,
        **kwargs,
    )


def _validation_dirs(tmp_path: Path, n_rows: int = 2 * BATCH) -> dict[str, Path]:
    return {
        language: _write_cache(
            tmp_path / "validation" / language,
            torch.randn(n_rows, D_IN, generator=torch.Generator().manual_seed(seed)),
            language=language,
        )
        for seed, language in enumerate(("de", "en"))
    }


def test_streaming_step_matches_one_cached_training_step(tmp_path, monkeypatch):
    monkeypatch.setattr("sae_pipeline.sae.train.plot_training_curves", lambda **_: None)
    x = torch.randn(BATCH, D_IN, generator=torch.Generator().manual_seed(0)) * 2.0
    cfg = _sae_cfg(_warm_start(tmp_path / "warm.safetensors"))

    cached = train_sae(
        cfg=cfg,
        cache_dir=_write_cache(tmp_path / "cache", x),
        d_in=D_IN,
        arch="jumprelu",
        d_sae=D_SAE,
        l0_target=L0_TARGET,
        out_dir=tmp_path / "cached",
        device="cpu",
    )
    cached_row = json.loads((tmp_path / "cached" / "train_log.jsonl").read_text().splitlines()[0])

    site = _site(tmp_path, cfg)
    entry = streaming_step(site, x, step=1)
    site.close()

    assert entry is not None
    for key in ("loss", "reconstruction_loss", "l0_penalty", "inverse_simpson_penalty", "lr", "hard_l0"):
        assert getattr(entry, key) == pytest.approx(cached_row[key], rel=1e-5, abs=1e-7), key
    cached_weights = load_file(str(cached))
    for key, value in site.sae.state_dict().items():
        torch.testing.assert_close(value, cached_weights[key], rtol=1e-5, atol=1e-6)


def test_streaming_step_logs_only_every_log_every_steps(tmp_path):
    site = _site(tmp_path, _sae_cfg(_warm_start(tmp_path / "warm.safetensors"), log_every=2))
    x = torch.randn(BATCH, D_IN)

    assert streaming_step(site, x, step=1) is None
    assert streaming_step(site, x, step=2) is not None
    site.close()


def test_streaming_step_applies_the_early_stop_lr_scale(tmp_path):
    site = _site(tmp_path, _sae_cfg(_warm_start(tmp_path / "warm.safetensors"), warmup_steps=0))
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
        _site(tmp_path, _sae_cfg(_warm_start(tmp_path / "warm.safetensors"), **overrides))


def test_validation_objective_is_the_training_loss_on_held_out_batches(tmp_path):
    validation_dirs = _validation_dirs(tmp_path)
    cfg = _sae_cfg(_warm_start(tmp_path / "warm.safetensors"), l0_warmup_steps=50)
    site = _site(tmp_path, cfg, validation_dirs, probe_tokens=2 * BATCH)

    evaluation = evaluate_site(site, "cpu")
    site.close()

    full_strength = coefficients_at(cfg, 50)
    for language, path in validation_dirs.items():
        rows = torch.cat(list(ShardReader(path).iter_shards())).to(torch.float32)
        expected = sum(
            float(training_loss(site.sae, batch, cfg, full_strength, target_l0=L0_TARGET).total.detach())
            for batch in rows.split(BATCH)
        ) / 2
        assert evaluation.objective[language] == pytest.approx(expected, rel=1e-5)
    assert evaluation.mixed_objective == pytest.approx(sum(evaluation.objective.values()) / 2)


def test_evaluation_logs_absolute_and_baseline_relative_metrics_and_saves_best(tmp_path):
    site = _site(tmp_path, _sae_cfg(_warm_start(tmp_path / "warm.safetensors")), _validation_dirs(tmp_path))
    record_baseline(site, "cpu")
    streaming_step(site, torch.randn(BATCH, D_IN), step=1)
    site.step = 1

    should_stop = evaluate_and_update(
        site, 1, "cpu", EarlyStoppingCfg(enabled=True, mode="objective", minimum_steps=0, patience=4)
    )
    site.close()

    assert should_stop is False
    rows = [json.loads(line) for line in site.validation_log_path.read_text().splitlines()]
    assert {row["language"] for row in rows} == {"de", "en"}
    for row in rows:
        assert row["is_best"] is True
        assert row["delta_fvu_vs_initial"] == pytest.approx(row["fvu"] - row["baseline_fvu"])
        assert row["delta_objective_vs_initial"] == pytest.approx(
            row["validation_objective"] - row["baseline_objective"]
        )
    assert (site.out_dir / "baseline_metrics.json").is_file()
    assert json.loads((site.out_dir / "best" / "step.json").read_text()) == {"step": 1, "tokens": BATCH}


def test_a_resumed_site_without_a_stored_baseline_reports_no_deltas(tmp_path):
    site = _site(tmp_path, _sae_cfg(_warm_start(tmp_path / "warm.safetensors")), _validation_dirs(tmp_path))
    site.step = 10

    record_baseline(site, "cpu")
    site.close()

    assert site.baseline is None
    assert not (site.out_dir / "baseline_metrics.json").exists()
