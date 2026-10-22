"""Streaming SAE evaluation: validation objective, baselines, divergence and early stopping."""

import json
import math

import pytest
import torch

from sae_pipeline.cache.reader import ShardReader
from sae_pipeline.config import EarlyStoppingCfg, StreamCfg
from sae_pipeline.sae.checkpoint_source import resolve_checkpoint
from sae_pipeline.sae.objective import coefficients_at, training_loss
from sae_pipeline.sae.train_streaming import evaluate_and_update, evaluate_site, record_baseline, streaming_step, write_checkpoint
from tests.sae.streaming_fakes import (
    BATCH,
    D_IN,
    L0_TARGET,
    CountingStream,
    make_sae_cfg,
    make_site,
    make_validation_dirs,
    patch_evaluations,
    run_training,
    write_warm_start,
)


def test_validation_objective_is_the_reconstruction_loss_without_penalties(tmp_path):
    validation_dirs = make_validation_dirs(tmp_path)
    cfg = make_sae_cfg(
        write_warm_start(tmp_path / "warm.safetensors"),
        l0_warmup_steps=50,
        inverse_simpson_floor=0.5,
        inverse_simpson_max_scale=1.0,
        auxk_scale=1.0,
    )
    site = make_site(tmp_path, cfg, validation_dirs, probe_tokens=2 * BATCH)

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
    site = make_site(tmp_path, make_sae_cfg(write_warm_start(tmp_path / "warm.safetensors"), warmup_steps=0, l0_warmup_steps=0), make_validation_dirs(tmp_path))
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


@pytest.mark.parametrize("overrides,minimum,enabled", [
    ({"warmup_steps": 3}, 3, False),
    ({"l0_warmup_steps": 5}, 5, False),
    ({"inverse_simpson_warmup_steps": 7}, 7, False),
    ({"l0_target_warmup_steps": 9}, 9, False),
    ({}, 11, True),
])
def test_best_selection_waits_for_all_warmups_and_starts_a_fresh_baseline(tmp_path, monkeypatch, overrides, minimum, enabled):
    site = make_site(tmp_path, make_sae_cfg(write_warm_start(tmp_path / "warm.safetensors"), **overrides), make_validation_dirs(tmp_path))
    cfg = EarlyStoppingCfg(enabled=enabled, minimum_steps=minimum if enabled else 0,
                           divergence_l0_factor=None, divergence_objective_factor=None)
    patch_evaluations(monkeypatch, objective=1.0)
    assert evaluate_and_update(site, minimum-1, "cpu", cfg) is None
    assert site.early_stop.best is None
    assert site.early_stop.checks_since_improve == 0
    assert not (site.out_dir / "best").exists()
    patch_evaluations(monkeypatch, objective=2.0)
    site.step = minimum
    assert evaluate_and_update(site, minimum, "cpu", cfg) is None
    assert site.early_stop.best == 2.0
    assert json.loads((site.out_dir / "best/step.json").read_text())["step"] == minimum
    site.close()
    rows = [json.loads(s) for s in site.validation_log_path.read_text().splitlines()]
    assert rows[0]["selection_eligible"] is False
    assert rows[-1]["selection_eligible"] is True


def test_resume_discards_legacy_best_without_post_warmup_provenance(tmp_path):
    cfg = make_sae_cfg(write_warm_start(tmp_path / "warm.safetensors"))
    site = make_site(tmp_path, cfg)
    site.early_stop.best, site.early_stop.checks_since_improve = 0.001, 5
    write_checkpoint(site, 3)
    site.close()
    checkpoint = resolve_checkpoint(site.out_dir, require_trainer_state=True)
    state = torch.load(checkpoint.trainer_state_path, weights_only=False)
    del state["best_selection_policy"]
    torch.save(state, checkpoint.trainer_state_path)
    resumed = make_site(tmp_path, cfg, resume_checkpoint=checkpoint)
    assert resumed.step == 3
    assert resumed.early_stop.best is None
    assert resumed.early_stop.checks_since_improve == 0
    resumed.close()


def test_reconstruction_selection_exports_the_lowest_objective_instead_of_best(tmp_path, monkeypatch):
    cfg = make_sae_cfg(write_warm_start(tmp_path / "warm.safetensors"), track_reconstruction_selection=True,
                   warmup_steps=0, l0_warmup_steps=0)
    site = make_site(tmp_path, cfg, make_validation_dirs(tmp_path))
    early_stopping = EarlyStoppingCfg(divergence_l0_factor=None, divergence_objective_factor=None)
    for step, objective in enumerate((2.0, 1.0, 3.0)):
        patch_evaluations(monkeypatch, objective=objective)
        site.step = step
        if step == 0:
            record_baseline(site, "cpu")
        else:
            evaluate_and_update(site, step, "cpu", early_stopping)
    site.close()

    state = json.loads((site.out_dir / "selection" / "selection.json").read_text())
    assert state["best"]["step"] == 1 and state["latest"]["step"] == 2
    assert (site.out_dir / state["best"]["export"]).exists()
    assert not (site.out_dir / "best").exists()


def test_a_resumed_site_without_a_stored_baseline_reports_no_deltas(tmp_path):
    site = make_site(tmp_path, make_sae_cfg(write_warm_start(tmp_path / "warm.safetensors")), make_validation_dirs(tmp_path))
    site.step = 10

    record_baseline(site, "cpu")
    site.close()

    assert site.baseline is None
    assert not (site.out_dir / "baseline_metrics.json").exists()


def test_a_checkpoint_resumes_on_new_validation_caches_only_when_asked(tmp_path):
    cfg = make_sae_cfg(write_warm_start(tmp_path / "warm.safetensors"))
    site = make_site(tmp_path, cfg, make_validation_dirs(tmp_path / "old"))
    record_baseline(site, "cpu")
    site.step = 10
    site.early_stop.best, site.early_stop.checks_since_improve, site.early_stop.lr_scale = 0.5, 4, 0.25
    write_checkpoint(site, site.step)
    site.close()
    checkpoint = resolve_checkpoint(site.out_dir, require_trainer_state=True)
    new_dirs = make_validation_dirs(tmp_path / "new", n_rows=3 * BATCH)

    with pytest.raises(ValueError, match="validation caches differ"):
        make_site(tmp_path, cfg, new_dirs, resume_checkpoint=checkpoint)
    resumed = make_site(tmp_path, cfg, new_dirs, resume_checkpoint=checkpoint, new_validation=True)
    resumed.close()

    assert resumed.step == 10
    assert (resumed.early_stop.best, resumed.early_stop.checks_since_improve, resumed.early_stop.lr_scale) == (None, 0, 0.25)
    assert not (site.out_dir / "baseline_metrics.json").exists()
    assert (site.out_dir / "baseline_metrics_previous_validation.json").exists()


def test_boundary_metrics_include_off_interval_terminal_without_plateau_update(tmp_path):
    cfg = make_sae_cfg(write_warm_start(tmp_path / "warm.safetensors"), eval_interval_steps=100, lr=0.0)
    site = make_site(tmp_path, cfg, make_validation_dirs(tmp_path))
    early = EarlyStoppingCfg(enabled=True, mode="objective", minimum_steps=0, patience=3, lr_plateau_patience=1)
    done = run_training(tmp_path, site, CountingStream(), stream_cfg=StreamCfg(enabled=True, token_cap=3 * BATCH), early_stopping=early)
    site.close()
    assert done and site.step == 3
    assert site.early_stop.best is None and site.early_stop.checks_since_improve == 0
    initial = json.loads((site.out_dir / "initial_metrics.json").read_text())
    terminal = json.loads((site.out_dir / "terminal_metrics.json").read_text())
    assert initial["tokens"] == 0 and terminal["tokens"] == 3 * BATCH
    assert set(initial["metrics"]) == set(terminal["metrics"]) == {"de", "en"}
    assert all("l0" in row and "fvu" in row for row in terminal["metrics"].values())
    assert not site.validation_log_path.read_text()


def test_terminal_divergence_is_detected_even_between_scheduled_evaluations(tmp_path, monkeypatch):
    patch_evaluations(monkeypatch, l0=4 * L0_TARGET)
    cfg = make_sae_cfg(write_warm_start(tmp_path / "warm.safetensors"), eval_interval_steps=100, l0_warmup_steps=0)
    site = make_site(tmp_path, cfg, make_validation_dirs(tmp_path))
    run_training(tmp_path, site, CountingStream(), stream_cfg=StreamCfg(enabled=True, token_cap=3 * BATCH))
    site.close()
    terminal = json.loads((site.out_dir / "terminal_metrics.json").read_text())
    assert terminal["diverged"] and not terminal["healthy"]
    assert not list(site.out_dir.glob("*_final.safetensors"))


def test_stops_early_once_the_validation_objective_plateaus(tmp_path):
    cfg = make_sae_cfg(write_warm_start(tmp_path / "warm.safetensors"), lr=0.0, eval_interval_steps=1, warmup_steps=0, l0_warmup_steps=0)
    site = make_site(tmp_path, cfg, make_validation_dirs(tmp_path))
    early_stopping = EarlyStoppingCfg(
        enabled=True, mode="objective", minimum_steps=0, patience=2, lr_plateau_patience=1
    )

    done = run_training(tmp_path, site, CountingStream(), stream_cfg=StreamCfg(enabled=True), early_stopping=early_stopping)
    site.close()

    assert done is True and site.stopped
    assert site.step == 3  # best at step 1, stale at 2 and 3
    rows = [json.loads(line) for line in site.validation_log_path.read_text().splitlines()]
    assert rows[-1]["early_stop_triggered"] is True
    assert resolve_checkpoint(site.out_dir, require_trainer_state=True).step == 3


def test_a_site_whose_l0_runs_away_stops_without_a_final_export_even_without_early_stopping(tmp_path, monkeypatch):
    patch_evaluations(monkeypatch, l0=4 * L0_TARGET)
    cfg = make_sae_cfg(write_warm_start(tmp_path / "warm.safetensors"), eval_interval_steps=2, l0_warmup_steps=0)
    site = make_site(tmp_path, cfg, make_validation_dirs(tmp_path))

    done = run_training(tmp_path, site, CountingStream(), stream_cfg=StreamCfg(enabled=True, token_cap=10 * BATCH))
    site.close()

    assert done is True and site.stopped and site.step == 2
    rows = [json.loads(line) for line in site.validation_log_path.read_text().splitlines()]
    assert all(row["diverged"].startswith("validation L0 12.0 above 3x") for row in rows)
    assert not list(site.out_dir.glob("*_final.safetensors"))
    assert resolve_checkpoint(site.out_dir, require_trainer_state=True).step == 2


def test_l0_is_not_checked_during_the_l0_warmup(tmp_path, monkeypatch):
    patch_evaluations(monkeypatch, l0=4 * L0_TARGET)
    site = make_site(tmp_path, make_sae_cfg(write_warm_start(tmp_path / "warm.safetensors"), l0_warmup_steps=10), make_validation_dirs(tmp_path))

    assert evaluate_and_update(site, 2, "cpu", EarlyStoppingCfg(mode="objective", patience=4)) is None
    site.close()


@pytest.mark.parametrize("objective", [math.nan, 50.0])
def test_a_site_whose_objective_explodes_or_is_not_finite_diverges(tmp_path, monkeypatch, objective):
    site = make_site(tmp_path, make_sae_cfg(write_warm_start(tmp_path / "warm.safetensors")), make_validation_dirs(tmp_path))
    site.early_stop.best = 1.0
    patch_evaluations(monkeypatch, objective=objective)

    stop = evaluate_and_update(site, 2, "cpu", EarlyStoppingCfg(mode="objective", patience=4))
    site.close()

    assert stop.diverged and "validation objective" in stop.reason
    assert site.early_stop.best == 1.0


def test_disabled_early_stopping_keeps_learning_rate_on_plateau(tmp_path):
    site = make_site(tmp_path, make_sae_cfg(write_warm_start(tmp_path / "warm.safetensors")), make_validation_dirs(tmp_path))
    cfg = EarlyStoppingCfg(enabled=False, minimum_steps=0, lr_plateau_patience=1,
                           divergence_l0_factor=None, divergence_objective_factor=None)
    for step in range(1, 5):
        assert evaluate_and_update(site, step, "cpu", cfg) is None
    site.close()
    assert site.early_stop.lr_scale == 1.0
    assert site.early_stop.checks_since_improve == 0
