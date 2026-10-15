"""Streaming SAE training: per-step parity with cached training, evaluation, loop."""

import dataclasses
import itertools
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from sae_pipeline.cache.manifest import CacheManifest
from sae_pipeline.cache.reader import ShardReader
from sae_pipeline.cache.writer import ShardWriter
from sae_pipeline.config import EarlyStoppingCfg, SAECfg, StreamCfg
from sae_pipeline.extraction.base import ActivationBatch
from sae_pipeline.extraction.fills import FillQueue, FillStream, fill_dir, write_header
from sae_pipeline.sae.jumprelu import JumpReLUSAE
from sae_pipeline.sae.checkpoint import resolve_checkpoint
from sae_pipeline.sae.objective import coefficients_at, training_loss
from sae_pipeline.sae.train import train_sae
from sae_pipeline.sae.train_streaming import (
    SiteEvaluation,
    evaluate_and_update,
    evaluate_site,
    init_streaming_site,
    record_baseline,
    run_streaming_training,
    streaming_step,
    write_checkpoint,
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


def test_validation_objective_is_the_reconstruction_loss_without_penalties(tmp_path):
    validation_dirs = _validation_dirs(tmp_path)
    cfg = _sae_cfg(
        _warm_start(tmp_path / "warm.safetensors"),
        l0_warmup_steps=50,
        inverse_simpson_floor=0.5,
        inverse_simpson_max_scale=1.0,
        auxk_scale=1.0,
    )
    site = _site(tmp_path, cfg, validation_dirs, probe_tokens=2 * BATCH)

    evaluation = evaluate_site(site, "cpu")
    site.close()

    full_strength = coefficients_at(cfg, 50)
    for language, path in validation_dirs.items():
        rows = torch.cat(list(ShardReader(path).iter_shards())).to(torch.float32)
        expected = sum(
            float(training_loss(site.sae, batch, cfg, full_strength, target_l0=L0_TARGET).reconstruction.detach())
            for batch in rows.split(BATCH)
        ) / 2
        assert evaluation.objective[language] == pytest.approx(expected, rel=1e-5)
    assert evaluation.mixed_objective == pytest.approx(sum(evaluation.objective.values()) / 2)


def test_evaluation_logs_absolute_and_baseline_relative_metrics_and_saves_best(tmp_path):
    site = _site(tmp_path, _sae_cfg(_warm_start(tmp_path / "warm.safetensors")), _validation_dirs(tmp_path))
    record_baseline(site, "cpu")
    streaming_step(site, torch.randn(BATCH, D_IN), step=1)
    site.step = 1

    stop = evaluate_and_update(
        site, 1, "cpu", EarlyStoppingCfg(enabled=True, mode="objective", minimum_steps=0, patience=4)
    )
    site.close()

    assert stop is None
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


def test_a_checkpoint_resumes_on_new_validation_caches_only_when_asked(tmp_path):
    cfg = _sae_cfg(_warm_start(tmp_path / "warm.safetensors"))
    site = _site(tmp_path, cfg, _validation_dirs(tmp_path / "old"))
    record_baseline(site, "cpu")
    site.step = 10
    site.early_stop.best, site.early_stop.checks_since_improve, site.early_stop.lr_scale = 0.5, 4, 0.25
    write_checkpoint(site, site.step)
    site.close()
    checkpoint = resolve_checkpoint(site.out_dir, require_trainer_state=True)
    new_dirs = _validation_dirs(tmp_path / "new", n_rows=3 * BATCH)

    with pytest.raises(ValueError, match="validation caches differ"):
        _site(tmp_path, cfg, new_dirs, resume_checkpoint=checkpoint)
    resumed = _site(tmp_path, cfg, new_dirs, resume_checkpoint=checkpoint, new_validation=True)
    resumed.close()

    assert resumed.step == 10
    assert (resumed.early_stop.best, resumed.early_stop.checks_since_improve, resumed.early_stop.lr_scale) == (None, 0, 0.25)
    assert not (site.out_dir / "baseline_metrics.json").exists()
    assert (site.out_dir / "baseline_metrics_previous_validation.json").exists()


SEQ_LEN, TOKENS_PER_FWD = 8, 2 * BATCH


class _CountingStream:
    """Token batches whose first token is a global sequence index."""

    def __init__(self) -> None:
        self.position = 0
        self.first_rows: list[int] = []

    def next_batch(self, batch_size: int) -> torch.Tensor:
        rows = torch.arange(self.position, self.position + batch_size).unsqueeze(1).repeat(1, SEQ_LEN)
        self.first_rows.append(self.position)
        self.position += batch_size
        return rows

    def state_dict(self) -> dict[str, object]:
        return {"position": self.position}

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.position = int(state["position"])


class _SeededExtractor:
    """Deterministic activations derived from each batch's first token."""

    def extract(self, token_batch: torch.Tensor) -> ActivationBatch:
        generator = torch.Generator().manual_seed(int(token_batch[0, 0]))
        return ActivationBatch(token_batch, {"site": torch.randn(token_batch.numel(), D_IN, generator=generator)})


class _StallingStream(_CountingStream):
    """Runs dry after ``batches`` batches, as a fill stream whose producer stopped."""

    def __init__(self, batches: int) -> None:
        super().__init__()
        self.batches = batches

    def next_batch(self, batch_size: int) -> torch.Tensor:
        if len(self.first_rows) == self.batches:
            raise TimeoutError("no fill arrived")
        return super().next_batch(batch_size)


def _run(tmp_path: Path, site, stream, *, stream_cfg: StreamCfg, early_stopping=None, name=None) -> bool:
    return run_streaming_training(
        sites={"site": site},
        token_stream=stream,
        extractor=_SeededExtractor(),
        stream_cfg=stream_cfg,
        early_stopping=early_stopping or EarlyStoppingCfg(),
        tokens_per_fwd=TOKENS_PER_FWD,
        seq_len=SEQ_LEN,
        device="cpu",
        out_root=tmp_path,
        name=name,
    )


def _logged_steps(site) -> list[int]:
    return [json.loads(line)["step"] for line in site.log_path.read_text().splitlines()]


def test_runs_to_the_token_cap_with_scheduled_evaluations(tmp_path):
    cfg = _sae_cfg(_warm_start(tmp_path / "warm.safetensors"), eval_interval_steps=2, ckpt_every=2)
    site = _site(tmp_path, cfg, _validation_dirs(tmp_path))
    stream = _CountingStream()

    done = _run(tmp_path, site, stream, stream_cfg=StreamCfg(enabled=True, token_cap=6 * BATCH))
    site.close()

    assert done is True
    assert (tmp_path / "DONE").exists()
    assert site.step == 6 and site.stopped
    assert _logged_steps(site) == [1, 2, 3, 4, 5, 6]
    evaluated = [json.loads(line)["step"] for line in site.validation_log_path.read_text().splitlines()]
    assert evaluated == [2, 2, 4, 4, 6, 6]
    assert stream.position == 3 * TOKENS_PER_FWD // SEQ_LEN


def test_pauses_at_the_time_limit_and_resumes_without_repeating_tokens(tmp_path, monkeypatch):
    warm = _warm_start(tmp_path / "warm.safetensors")
    clock = itertools.count()
    monkeypatch.setattr(
        "sae_pipeline.sae.train_streaming.time", SimpleNamespace(monotonic=lambda: float(next(clock)))
    )
    site = _site(tmp_path, _sae_cfg(warm, ckpt_every=100))
    first = _CountingStream()

    done = _run(tmp_path, site, first, stream_cfg=StreamCfg(enabled=True, stop_after_seconds=3))
    site.close()

    assert done is False and not (tmp_path / "DONE").exists()
    assert site.step == 4  # two forward passes, each two SAE batches

    resumed = _site(
        tmp_path,
        _sae_cfg(warm, ckpt_every=100),
        resume_checkpoint=resolve_checkpoint(site.out_dir, require_trainer_state=True),
    )
    second = _CountingStream()
    done = _run(tmp_path, resumed, second, stream_cfg=StreamCfg(enabled=True, token_cap=8 * BATCH))
    resumed.close()

    assert done is True
    assert resumed.step == 8
    assert _logged_steps(resumed) == list(range(1, 9))
    assert set(first.first_rows).isdisjoint(second.first_rows)
    assert second.first_rows[0] == first.position


def test_a_named_trainer_pauses_when_its_stream_times_out_and_resumes_after_it(tmp_path):
    warm = _warm_start(tmp_path / "warm.safetensors")
    site = _site(tmp_path, _sae_cfg(warm, ckpt_every=100))
    first = _StallingStream(batches=2)

    done = _run(tmp_path, site, first, stream_cfg=StreamCfg(enabled=True), name="c0")
    site.close()

    assert done is False and site.step == 4
    assert (tmp_path / "stream_state_c0.pt").exists() and not (tmp_path / "stream_state.pt").exists()
    resumed = _site(
        tmp_path,
        _sae_cfg(warm, ckpt_every=100),
        resume_checkpoint=resolve_checkpoint(site.out_dir, require_trainer_state=True),
    )
    second = _CountingStream()
    done = _run(tmp_path, resumed, second, stream_cfg=StreamCfg(enabled=True, token_cap=6 * BATCH), name="c0")
    resumed.close()

    assert done is True and (tmp_path / "DONE_c0").exists() and not (tmp_path / "DONE").exists()
    assert second.first_rows[0] == first.position


def test_a_resumed_site_continues_the_inverse_simpson_window(tmp_path):
    cfg = _sae_cfg(_warm_start(tmp_path / "warm.safetensors"), inverse_simpson_window_steps=8)
    site = _site(tmp_path, cfg)
    for step in (1, 2):
        streaming_step(site, torch.randn(BATCH, D_IN, generator=torch.Generator().manual_seed(step)), step)
    write_checkpoint(site, 2)
    site.close()

    resumed = _site(tmp_path, cfg, resume_checkpoint=resolve_checkpoint(site.out_dir, require_trainer_state=True))
    resumed.close()

    assert site.inverse_simpson_usage.sum() > 0
    torch.testing.assert_close(resumed.inverse_simpson_usage, site.inverse_simpson_usage)


def test_a_resumed_site_continues_the_auxk_dead_counter(tmp_path):
    cfg = _sae_cfg(_warm_start(tmp_path / "warm.safetensors"), auxk_scale=1 / 32, auxk_k=2, auxk_dead_tokens=BATCH)
    site = _site(tmp_path, cfg)
    with torch.no_grad():
        site.sae.log_theta[:4] = 10.0  # four features that never fire
    for step in (1, 2):
        streaming_step(site, torch.randn(BATCH, D_IN, generator=torch.Generator().manual_seed(step)), step)
    write_checkpoint(site, 2)
    site.close()

    resumed = _site(tmp_path, cfg, resume_checkpoint=resolve_checkpoint(site.out_dir, require_trainer_state=True))
    resumed.close()

    assert bool((site.tokens_since_fired[:4] == 2 * BATCH).all())
    torch.testing.assert_close(resumed.tokens_since_fired, site.tokens_since_fired)


def test_streaming_resets_resample_unused_features_and_survive_a_resume(tmp_path):
    cfg = _sae_cfg(
        _warm_start(tmp_path / "warm.safetensors"),
        feature_use_strategy="residual_reset",
        residual_reset_start_step=2,
        residual_reset_every_steps=10,
        residual_reset_max_features=4,
        residual_reset_pool_size=BATCH,
        auxk_scale=1 / 32,
    )
    site = _site(tmp_path, cfg)
    with torch.no_grad():
        site.sae.log_theta[:4] = 10.0  # four features that never fire
    for step in (1, 2):
        streaming_step(site, torch.randn(BATCH, D_IN, generator=torch.Generator().manual_seed(step)), step)
    write_checkpoint(site, 2)
    site.close()

    resumed = _site(tmp_path, cfg, resume_checkpoint=resolve_checkpoint(site.out_dir, require_trainer_state=True))
    resumed.close()

    assert site.feature_resets == 4
    assert bool((site.sae.log_theta[:4] < 10.0).all())  # recalibrated to fire again
    assert bool((site.tokens_since_fired[:4] == 0).all())  # a replaced latent starts a new history
    assert (resumed.feature_resets, resumed.last_reset_step) == (4, 2)


def test_the_log_row_tells_why_auxk_dead_latents_are_inactive(tmp_path):
    cfg = _sae_cfg(_warm_start(tmp_path / "warm.safetensors"), auxk_scale=1 / 32, auxk_k=2, auxk_dead_tokens=BATCH)
    site = _site(tmp_path, cfg)
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


def test_stops_early_once_the_validation_objective_plateaus(tmp_path):
    cfg = _sae_cfg(_warm_start(tmp_path / "warm.safetensors"), lr=0.0, eval_interval_steps=1)
    site = _site(tmp_path, cfg, _validation_dirs(tmp_path))
    early_stopping = EarlyStoppingCfg(
        enabled=True, mode="objective", minimum_steps=0, patience=2, lr_plateau_patience=1
    )

    done = _run(tmp_path, site, _CountingStream(), stream_cfg=StreamCfg(enabled=True), early_stopping=early_stopping)
    site.close()

    assert done is True and site.stopped
    assert site.step == 3  # best at step 1, stale at 2 and 3
    rows = [json.loads(line) for line in site.validation_log_path.read_text().splitlines()]
    assert rows[-1]["early_stop_triggered"] is True
    assert resolve_checkpoint(site.out_dir, require_trainer_state=True).step == 3


def _evaluations(monkeypatch, *, l0=None, objective=None):
    """Make every evaluation report this L0 and/or objective."""
    real = evaluate_site

    def changed(site, device):
        evaluation = real(site, device)
        metrics = {k: dataclasses.replace(m, l0=m.l0 if l0 is None else l0) for k, m in evaluation.metrics.items()}
        objectives = {k: v if objective is None else objective for k, v in evaluation.objective.items()}
        return SiteEvaluation(metrics, objectives)

    monkeypatch.setattr("sae_pipeline.sae.train_streaming.evaluate_site", changed)


def test_a_site_whose_l0_runs_away_stops_without_a_final_export_even_without_early_stopping(tmp_path, monkeypatch):
    _evaluations(monkeypatch, l0=4 * L0_TARGET)
    cfg = _sae_cfg(_warm_start(tmp_path / "warm.safetensors"), eval_interval_steps=2, l0_warmup_steps=0)
    site = _site(tmp_path, cfg, _validation_dirs(tmp_path))

    done = _run(tmp_path, site, _CountingStream(), stream_cfg=StreamCfg(enabled=True, token_cap=10 * BATCH))
    site.close()

    assert done is True and site.stopped and site.step == 2
    rows = [json.loads(line) for line in site.validation_log_path.read_text().splitlines()]
    assert all(row["diverged"].startswith("validation L0 12.0 above 3x") for row in rows)
    assert not list(site.out_dir.glob("*_final.safetensors"))
    assert resolve_checkpoint(site.out_dir, require_trainer_state=True).step == 2


def test_l0_is_not_checked_during_the_l0_warmup(tmp_path, monkeypatch):
    _evaluations(monkeypatch, l0=4 * L0_TARGET)
    site = _site(tmp_path, _sae_cfg(_warm_start(tmp_path / "warm.safetensors"), l0_warmup_steps=10), _validation_dirs(tmp_path))

    assert evaluate_and_update(site, 2, "cpu", EarlyStoppingCfg(mode="objective", patience=4)) is None
    site.close()


@pytest.mark.parametrize("objective", [math.nan, 50.0])
def test_a_site_whose_objective_explodes_or_is_not_finite_diverges(tmp_path, monkeypatch, objective):
    site = _site(tmp_path, _sae_cfg(_warm_start(tmp_path / "warm.safetensors")), _validation_dirs(tmp_path))
    site.early_stop.best = 1.0
    _evaluations(monkeypatch, objective=objective)

    stop = evaluate_and_update(site, 2, "cpu", EarlyStoppingCfg(mode="objective", patience=4))
    site.close()

    assert stop.diverged and "validation objective" in stop.reason
    assert site.early_stop.best == 1.0


def test_two_consumers_train_their_own_sites_from_one_producers_fills(tmp_path):
    fills = tmp_path / "fills"
    write_header(fills, {"model": "m", "seq_len": SEQ_LEN, "slugs": ["a", "b"]})
    generator = torch.Generator().manual_seed(0)
    for index in range(3):
        partial = fills / f"fill_{index:06d}.partial"
        partial.mkdir()
        save_file({"token_ids": torch.zeros(6, SEQ_LEN, dtype=torch.long)}, str(partial / "token_ids.safetensors"))
        for slug in ("a", "b"):
            save_file({"activations": torch.randn(6 * SEQ_LEN, D_IN, generator=generator)}, str(partial / f"{slug}.safetensors"))
        partial.rename(fill_dir(fills, index))
    warm = _warm_start(tmp_path / "warm.safetensors")
    run = tmp_path / "run"

    for name, slug in (("c0", "a"), ("c1", "b")):
        site = init_streaming_site(
            slug=slug, sae_cfg=_sae_cfg(warm), d_in=D_IN, arch="jumprelu", d_sae=D_SAE, l0_target=L0_TARGET,
            out_dir=run / slug, device="cpu", validation_cache_dirs={}, probe_tokens=BATCH,
        )
        stream = FillStream(fills, name, [slug], header={"model": "m", "seq_len": SEQ_LEN}, device="cpu", poll_seconds=0)
        done = run_streaming_training(
            sites={slug: site}, token_stream=stream, extractor=stream,
            stream_cfg=StreamCfg(enabled=True, token_cap=4 * BATCH), early_stopping=EarlyStoppingCfg(),
            tokens_per_fwd=TOKENS_PER_FWD, seq_len=SEQ_LEN, device="cpu", out_root=run, name=name,
        )
        stream.close()
        site.close()
        assert done and site.step == 4
        # Two forward passes of 8 sequences: fills 0 and 1, and 4 sequences of fill 2.
        assert stream.state_dict() == {"fill": 2, "sequence": 4}

    assert sorted(p.name for p in run.iterdir()) == ["DONE_c0", "DONE_c1", "a", "b", "stream_state_c0.pt", "stream_state_c1.pt"]
    assert FillQueue(fills, ["c0", "c1"]).release() == 2


def test_disabled_early_stopping_keeps_learning_rate_on_plateau(tmp_path):
    site = _site(tmp_path, _sae_cfg(_warm_start(tmp_path / "warm.safetensors")), _validation_dirs(tmp_path))
    cfg = EarlyStoppingCfg(enabled=False, minimum_steps=0, lr_plateau_patience=1,
                           divergence_l0_factor=None, divergence_objective_factor=None)
    for step in range(1, 5):
        assert evaluate_and_update(site, step, "cpu", cfg) is None
    site.close()
    assert site.early_stop.lr_scale == 1.0
    assert site.early_stop.checks_since_improve == 0


def test_streaming_seed_and_target_schedule_match_cached_recipe(tmp_path):
    cfg = _sae_cfg(_warm_start(tmp_path / "warm.safetensors"), seed=123,
                   l0_target_start=9, l0_target_warmup_steps=4)
    site = _site(tmp_path, cfg)
    first_random = torch.rand(5)
    x = torch.randn(BATCH, D_IN, generator=torch.Generator().manual_seed(0))
    entry = streaming_step(site, x, 1)
    site.close()
    torch.manual_seed(999)
    again = _site(tmp_path, cfg)
    torch.testing.assert_close(torch.rand(5), first_random)
    again.close()
    assert entry.target_l0 == 7.5


def test_cached_training_rejects_unimplemented_objective_stop_before_io(tmp_path):
    with pytest.raises(ValueError, match="stability"):
        train_sae(cfg=SAECfg(), cache_dir=tmp_path / "missing", d_in=D_IN,
                  arch="jumprelu", d_sae=D_SAE, l0_target=L0_TARGET, out_dir=tmp_path / "out",
                  early_stopping=EarlyStoppingCfg(enabled=True, mode="objective", patience=4))
    assert not (tmp_path / "out").exists()


def test_normalized_streaming_checkpoints_and_best_carry_metadata(tmp_path):
    from sae_pipeline.sae.normalization import WholeVectorNormalization, write_normalization
    warm = _warm_start(tmp_path / "warm.safetensors")
    metadata = tmp_path / "norm"
    metadata.mkdir()
    statistic = WholeVectorNormalization(0.5, 4.0, 64, D_IN)
    write_normalization(metadata / "normalization.json", statistic)
    cfg = _sae_cfg(warm, input_normalization="whole_vector")
    validation = _validation_dirs(tmp_path)
    site = _site(tmp_path, cfg, validation, normalization_dir=metadata)
    evaluate_and_update(site, 0, "cpu", EarlyStoppingCfg(divergence_l0_factor=None))
    write_checkpoint(site, 0)
    site.close()
    expected = json.loads((metadata / "normalization.json").read_text())
    assert json.loads((site.out_dir / "normalization.json").read_text()) == expected
    assert json.loads((site.out_dir / "best" / "normalization.json").read_text()) == expected
    resumed = _site(tmp_path, cfg, validation, resume_checkpoint=resolve_checkpoint(site.out_dir, require_trainer_state=True))
    assert resumed.normalization_scale == 0.5
    resumed.close()
    write_normalization(metadata / "normalization.json", dataclasses.replace(statistic, input_scale=0.25))
    with pytest.raises(ValueError, match="normalization scale differs"):
        _site(tmp_path, cfg, validation, normalization_dir=metadata,
              resume_checkpoint=resolve_checkpoint(site.out_dir, require_trainer_state=True))
    assert json.loads((site.out_dir / "normalization.json").read_text()) == expected


def test_token_cap_stops_inside_forward_and_requires_batch_alignment(tmp_path, monkeypatch):
    monkeypatch.setattr("sae_pipeline.sae.train_streaming.refresh_plot", lambda *a, **kw: None)
    warm = _warm_start(tmp_path / "warm.safetensors")
    site = _site(tmp_path, _sae_cfg(warm, ckpt_every=3))
    with pytest.raises(ValueError, match="multiple"):
        _run(tmp_path, site, _CountingStream(), stream_cfg=StreamCfg(token_cap=BATCH + 1))
    assert site.step == 0
    assert _run(tmp_path, site, _CountingStream(), stream_cfg=StreamCfg(token_cap=3 * BATCH))
    site.close()
    assert site.step == 3 and site.tokens_seen() == 3 * BATCH


def test_periodic_snapshot_is_taken_after_all_batches_in_forward(tmp_path, monkeypatch):
    from sae_pipeline.sae.checkpoint import load_trainer_state
    from sae_pipeline.sae.train_streaming import committed_stream_checkpoints
    monkeypatch.setattr("sae_pipeline.sae.train_streaming.refresh_plot", lambda *a, **kw: None)
    site = _site(tmp_path, _sae_cfg(_warm_start(tmp_path / "warm.safetensors"), ckpt_every=3))
    # The periodic save is triggered at step 3 inside forward 2, but must contain step 4.
    assert not _run(tmp_path, site, _StallingStream(2), stream_cfg=StreamCfg())
    site.close()
    committed = committed_stream_checkpoints(tmp_path, {"site": site.out_dir})["site"]
    state = load_trainer_state(tmp_path / "stream_state.pt")
    assert committed.step == 4
    assert state["tokens_processed"] == 4 * BATCH


@pytest.mark.parametrize("failure_point", ["second_site", "cursor"])
def test_interrupted_multi_site_snapshot_keeps_previous_pairs_and_resumes_exactly(tmp_path, monkeypatch, failure_point):
    import sae_pipeline.sae.train_streaming as module
    monkeypatch.setattr(module, "refresh_plot", lambda *a, **kw: None)
    warm = _warm_start(tmp_path / "warm.safetensors")
    cfg = _sae_cfg(warm, ckpt_every=3, keep_last_checkpoints=1)
    class MultiExtractor:
        def extract(self, tokens):
            batch = _SeededExtractor().extract(tokens)
            return ActivationBatch(tokens, {slug: batch.activations["site"] for slug in ("a", "b")})
    def make_sites(root, resume=False):
        selected = module.committed_stream_checkpoints(root, {s: root / s for s in ("a", "b")}) if resume else {}
        return {s: init_streaming_site(slug=s, sae_cfg=cfg, d_in=D_IN, arch="jumprelu", d_sae=D_SAE,
                    l0_target=L0_TARGET, out_dir=root / s, device="cpu", validation_cache_dirs={},
                    probe_tokens=BATCH, resume_checkpoint=selected.get(s)) for s in ("a", "b")}
    def run(root, sites, stream, cap=None):
        return run_streaming_training(sites=sites, token_stream=stream, extractor=MultiExtractor(),
                stream_cfg=StreamCfg(token_cap=cap), early_stopping=EarlyStoppingCfg(),
                tokens_per_fwd=TOKENS_PER_FWD, seq_len=SEQ_LEN, device="cpu", out_root=root)
    root = tmp_path / "interrupted"
    sites = make_sites(root)
    assert not run(root, sites, _StallingStream(2))
    for site in sites.values(): site.close()
    old = module.committed_stream_checkpoints(root, {s: root / s for s in sites})
    sites = make_sites(root, resume=True)
    with monkeypatch.context() as fault:
        if failure_point == "second_site":
            actual = module.write_checkpoint
            def failing(site, step, **kwargs):
                if site.slug == "b": raise RuntimeError("interrupted transaction")
                return actual(site, step, **kwargs)
            fault.setattr(module, "write_checkpoint", failing)
        else:
            fault.setattr(module, "_save_stream_state", lambda *a: (_ for _ in ()).throw(RuntimeError("interrupted transaction")))
        with pytest.raises(RuntimeError, match="interrupted transaction"):
            run(root, sites, _CountingStream(), 8 * BATCH)
    for site in sites.values(): site.close()
    for checkpoint in old.values():
        assert checkpoint.weights_path.exists() and checkpoint.trainer_state_path.exists()
    assert module.committed_stream_checkpoints(root, {s: root / s for s in sites}) == old
    resumed = make_sites(root, resume=True)
    assert run(root, resumed, _CountingStream(), 8 * BATCH)
    reference_root = tmp_path / "reference"
    reference = make_sites(reference_root)
    assert run(reference_root, reference, _CountingStream(), 8 * BATCH)
    for slug in resumed:
        resumed[slug].close(); reference[slug].close()
        for key, weights in reference[slug].sae.state_dict().items():
            torch.testing.assert_close(resumed[slug].sae.state_dict()[key], weights, rtol=0, atol=0)
        assert _logged_steps(resumed[slug]) == list(range(1, 9))
