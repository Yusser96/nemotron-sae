"""Full held-out training loss, independent replay statistics and plateau persistence."""

import copy
import random
import numpy as np
import pytest
import torch
from types import SimpleNamespace
from tests.sae import streaming_fakes as f

h = SimpleNamespace(
    _sae_cfg=f.make_sae_cfg,
    _warm_start=f.write_warm_start,
    _site=f.make_site,
    _validation_dirs=f.make_validation_dirs,
    BATCH=f.BATCH,
    D_SAE=f.D_SAE,
)
from sae_pipeline.config import EarlyStoppingCfg
from sae_pipeline.sae.checkpoint_source import resolve_checkpoint
from sae_pipeline.sae.objective import coefficients_at, training_loss
from sae_pipeline.sae.train_streaming import (
    evaluate_site,
    evaluate_and_update,
    write_checkpoint,
    record_baseline,
)
from sae_pipeline.sae.validation_stopping import configure_stopping, eligibility_step


def test_full_objective_matches_training_terms_and_does_not_mutate_state(tmp_path):
    cfg = h._sae_cfg(
        h._warm_start(tmp_path / "warm.safetensors"), auxk_scale=0.3, auxk_dead_tokens=1
    )
    site = h._site(
        tmp_path, cfg, h._validation_dirs(tmp_path), probe_tokens=2 * h.BATCH
    )
    site.step = 10
    site.tokens_since_fired.fill_(1000)
    site.inverse_simpson_usage.copy_(torch.linspace(0, 1, h.D_SAE))
    tensors = {
        name: getattr(site, name).clone()
        for name in [
            "tokens_since_fired",
            "inverse_simpson_usage",
            "ever_fired",
            "feature_frequency",
        ]
    }
    weights = {k: v.clone() for k, v in site.sae.state_dict().items()}
    optim = copy.deepcopy(site.optim.state_dict())
    rng = torch.get_rng_state().clone()
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    site.sae.eval()
    got = evaluate_site(site, "cpu")
    assert not site.sae.training
    assert torch.equal(rng, torch.get_rng_state()) and random.getstate() == python_rng
    assert np.array_equal(numpy_rng[1], np.random.get_state()[1])
    assert optim == site.optim.state_dict()
    for name, value in tensors.items():
        assert torch.equal(value, getattr(site, name))
    for key, value in weights.items():
        assert torch.equal(value, site.sae.state_dict()[key])
    coefficients = coefficients_at(cfg, site.step)
    for language, reader in site.validation_readers.items():
        usage = tensors["inverse_simpson_usage"].clone()
        dead = tensors["tokens_since_fired"].clone()
        total = rows = 0
        from sae_pipeline.sae.validation import validation_batches

        for x in validation_batches(
            reader,
            n_tokens=site.probe_tokens,
            batch_size=cfg.batch_size,
            input_scale=site.normalization_scale,
            input_center=None,
            device="cpu",
        ):
            with torch.no_grad():
                terms = training_loss(
                    site.sae,
                    x,
                    cfg,
                    coefficients,
                    target_l0=float(site.l0_target),
                    usage=usage,
                    tokens_since_fired=dead,
                )
            total += float(terms.total) * len(x)
            rows += len(x)
        assert got.training_objective[language] == pytest.approx(total / rows)
        c = got.loss_components[language]
        assert c["auxk_weighted"] > 0
        assert c["total"] == pytest.approx(
            c["reconstruction"]
            + c["sparsity_weighted"]
            + c["is_weighted"]
            + c["auxk_weighted"]
        )
    assert got.stopping_score("full_training_loss") == pytest.approx(
        sum(got.training_objective.values()) / 2
    )
    site.close()


def test_all_warmups_protect_baseline_patience_and_lr(tmp_path, monkeypatch):
    cfg = h._sae_cfg(
        h._warm_start(tmp_path / "warm.safetensors"),
        warmup_steps=2,
        l0_warmup_steps=4,
        l0_target_start=1,
        l0_target_warmup_steps=6,
        inverse_simpson_warmup_steps=8,
    )
    site = h._site(tmp_path, cfg, h._validation_dirs(tmp_path))
    measured = evaluate_site(site, "cpu")
    measured.training_objective = {"en": 3.0, "de": 5.0}
    monkeypatch.setattr(
        "sae_pipeline.sae.train_streaming.evaluate_site", lambda *args: measured
    )
    rule = EarlyStoppingCfg(
        enabled=True,
        mode="objective",
        score="full_training_loss",
        minimum_steps=0,
        patience=4,
        min_lr_scale=1.0,
        divergence_l0_factor=None,
        divergence_objective_factor=None,
    )
    assert eligibility_step(cfg, rule) == 8
    for step in range(1, 8):
        site.step = step
        assert evaluate_and_update(site, step, "cpu", rule) is None
        assert (
            site.early_stop.best is None and site.early_stop.checks_since_improve == 0
        )
    site.step = 8
    assert evaluate_and_update(site, 8, "cpu", rule) is None
    assert site.early_stop.best == 4.0 and site.early_stop.checks_since_improve == 0
    best = site.stopping_state["best_full_objective"]
    assert (site.out_dir / best["export"]).exists()
    site.early_stop.lr_scale = 0.5
    configure_stopping(site, rule)
    assert site.early_stop.lr_scale == 1.0
    path = write_checkpoint(site, 8)
    site.close()
    resumed = h._site(
        tmp_path,
        cfg,
        {lang: tmp_path / "validation" / lang for lang in ("en", "de")},
        resume_checkpoint=resolve_checkpoint(
            site.out_dir, filename=path.name, require_trainer_state=True
        ),
    )
    configure_stopping(resumed, rule)
    assert (
        resumed.stopping_state == site.stopping_state and resumed.early_stop.best == 4.0
    )
    for step in range(9, 12):
        resumed.step = step
        assert evaluate_and_update(resumed, step, "cpu", rule) is None
    resumed.step = 12
    assert evaluate_and_update(resumed, 12, "cpu", rule).reason == "stopped improving"
    assert resumed.early_stop.lr_scale == 1.0
    site.close()
    resumed.close()


def test_reconstruction_score_migrates_once_without_resetting_progress(tmp_path):
    cfg = h._sae_cfg(h._warm_start(tmp_path / "warm.safetensors"))
    site = h._site(tmp_path, cfg)
    site.step = 52
    site.early_stop.best = 0.1
    site.early_stop.checks_since_improve = 9
    before = {k: v.clone() for k, v in site.sae.state_dict().items()}
    rule = EarlyStoppingCfg(
        mode="objective", score="full_training_loss", patience=10, min_lr_scale=1
    )
    configure_stopping(site, rule)
    assert (
        site.step == 52
        and site.early_stop.best is None
        and site.early_stop.checks_since_improve == 0
    )
    site.early_stop.best = 3
    site.early_stop.checks_since_improve = 2
    configure_stopping(site, rule)
    assert site.early_stop.best == 3 and site.early_stop.checks_since_improve == 2
    for k, v in before.items():
        assert torch.equal(v, site.sae.state_dict()[k])
    site.close()


def test_best_full_objective_is_saved_during_warmup_independently_of_reconstruction(
    tmp_path, monkeypatch
):
    cfg = h._sae_cfg(
        h._warm_start(tmp_path / "warm.safetensors"),
        track_reconstruction_selection=True,
    )
    site = h._site(tmp_path, cfg, h._validation_dirs(tmp_path))
    evaluation = evaluate_site(site, "cpu")
    evaluation.objective = {"en": 3.0, "de": 3.0}
    evaluation.training_objective = {"en": 6.0, "de": 6.0}
    monkeypatch.setattr(
        "sae_pipeline.sae.train_streaming.evaluate_site", lambda *args: evaluation
    )
    rule = EarlyStoppingCfg(
        enabled=True,
        mode="objective",
        score="full_training_loss",
        minimum_steps=50_000,
        patience=10,
        min_lr_scale=1,
        divergence_l0_factor=None,
        divergence_objective_factor=None,
    )
    configure_stopping(site, rule)
    record_baseline(site, "cpu")
    assert site.stopping_state["best_full_objective"]["step"] == 0
    site.step = 1
    evaluation.objective = {"en": 2.0, "de": 2.0}
    evaluation.training_objective = {"en": 10.0, "de": 10.0}
    assert evaluate_and_update(site, 1, "cpu", rule) is None
    assert site.selection_state["best"]["step"] == 1
    assert site.stopping_state["best_full_objective"]["step"] == 0
    site.step = 2
    evaluation.objective = {"en": 4.0, "de": 4.0}
    evaluation.training_objective = {"en": 5.0, "de": 5.0}
    assert evaluate_and_update(site, 2, "cpu", rule) is None
    assert site.selection_state["best"]["step"] == 1
    assert site.stopping_state["best_full_objective"]["step"] == 2
    assert site.early_stop.best is None and site.early_stop.checks_since_improve == 0
    assert (
        site.out_dir / site.stopping_state["best_full_objective"]["export"]
    ).is_file()
    assert (site.out_dir / site.selection_state["best"]["export"]).is_file()
    site.close()
