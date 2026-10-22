"""Early stopping: objective plateaus with LR reduction, and bilingual metric stability."""

from types import SimpleNamespace

import pytest

from sae_pipeline.config import EarlyStoppingCfg
from sae_pipeline.sae.early_stopping import ObjectiveEarlyStop, StabilityEarlyStop


def _cfg(**overrides) -> EarlyStoppingCfg:
    defaults = dict(
        enabled=True,
        mode="objective",
        minimum_steps=0,
        patience=6,
        lr_plateau_patience=2,
        lr_reduce_factor=0.5,
        min_lr_scale=0.05,
        improvement_rel_tol=0.005,
    )
    defaults.update(overrides)
    return EarlyStoppingCfg(**defaults)


def test_ineligible_evaluations_track_best_but_never_count_as_stale():
    stop = ObjectiveEarlyStop()
    cfg = _cfg()

    assert stop.update(1.0, cfg, eligible=False) == (True, False, False)
    for _ in range(10):
        assert stop.update(1.0, cfg, eligible=False) == (False, False, False)
    assert stop.update(0.5, cfg, eligible=False) == (True, False, False)
    assert stop.checks_since_improve == 0
    assert stop.best == 0.5


def test_relative_improvement_threshold_gates_is_best():
    stop = ObjectiveEarlyStop()
    cfg = _cfg(improvement_rel_tol=0.01)
    stop.update(1.0, cfg)

    # 0.5% lower is below the 1% threshold: not an improvement.
    is_best, _, _ = stop.update(0.995, cfg)
    assert is_best is False
    assert stop.best == 1.0
    assert stop.checks_since_improve == 1

    # 2% lower clears the threshold.
    is_best, _, _ = stop.update(0.98, cfg)
    assert is_best is True
    assert stop.best == 0.98
    assert stop.checks_since_improve == 0


def test_lr_reduced_every_plateau_patience_then_stops_at_patience():
    stop = ObjectiveEarlyStop()
    cfg = _cfg(patience=6, lr_plateau_patience=2, lr_reduce_factor=0.5, min_lr_scale=0.1)
    stop.update(1.0, cfg)

    results = [stop.update(1.0, cfg) for _ in range(6)]
    lr_reductions = [lr_reduced for _, lr_reduced, _ in results]
    stops = [should_stop for _, _, should_stop in results]

    # Reductions at checks_since_improve == 2 and 4 (multiples of 2, below patience=6).
    assert lr_reductions == [False, True, False, True, False, False]
    assert stop.lr_scale == pytest.approx(0.25)
    # Stop fires once checks_since_improve reaches patience (the 6th stale check).
    assert stops == [False, False, False, False, False, True]


def test_lr_scale_floors_at_min_lr_scale():
    stop = ObjectiveEarlyStop()
    cfg = _cfg(patience=100, lr_plateau_patience=1, lr_reduce_factor=0.5, min_lr_scale=0.2)
    stop.update(1.0, cfg)
    for _ in range(10):
        stop.update(1.0, cfg)
    assert stop.lr_scale == pytest.approx(0.2)


def test_state_dict_round_trip():
    stop = ObjectiveEarlyStop()
    cfg = _cfg()
    stop.update(1.0, cfg)
    stop.update(1.0, cfg)

    restored = ObjectiveEarlyStop()
    restored.load_state_dict(stop.state_dict())
    assert restored == stop


def test_non_finite_objective_raises():
    with pytest.raises(FloatingPointError):
        ObjectiveEarlyStop().update(float("nan"), _cfg())


def test_bilingual_early_stop_requires_stable_fvu_and_target_l0():
    cfg = EarlyStoppingCfg(
        enabled=True,
        fvu_relative_tolerance=0.01,
        ev_absolute_tolerance=0.005,
        l0_relative_tolerance=0.10,
        patience=2,
    )
    monitor = StabilityEarlyStop()
    first = {lang: _validation_metric(10.0, 0.50) for lang in ("de", "en")}
    second = {"de": _validation_metric(10.1, 0.498), "en": _validation_metric(9.9, 0.498)}
    third = {"de": _validation_metric(10.0, 0.496), "en": _validation_metric(10.0, 0.496)}

    stable, l0_ok = monitor.update(first, target_l0=10.0, cfg=cfg)
    assert not stable and l0_ok and monitor.stable_intervals == 0
    stable, l0_ok = monitor.update(second, target_l0=10.0, cfg=cfg)
    assert stable and l0_ok and monitor.stable_intervals == 1
    stable, l0_ok = monitor.update(third, target_l0=10.0, cfg=cfg)
    assert stable and l0_ok and monitor.stable_intervals == 2


def test_bilingual_early_stop_resets_if_any_language_is_unstable_or_l0_misses():
    cfg = EarlyStoppingCfg(enabled=True)
    monitor = StabilityEarlyStop()
    monitor.update(
        {lang: _validation_metric(10.0, 0.50) for lang in ("de", "en")},
        target_l0=10.0,
        cfg=cfg,
    )
    stable, _ = monitor.update(
        {"de": _validation_metric(10.0, 0.499), "en": _validation_metric(10.0, 0.45)},
        target_l0=10.0,
        cfg=cfg,
    )
    assert not stable and monitor.stable_intervals == 0
    stable, l0_ok = monitor.update(
        {lang: _validation_metric(15.0, 0.499) for lang in ("de", "en")},
        target_l0=10.0,
        cfg=cfg,
    )
    assert not stable and not l0_ok and monitor.stable_intervals == 0


def test_early_stop_uses_post_warmup_evaluations_as_its_baseline():
    cfg = EarlyStoppingCfg(enabled=True, patience=2)
    monitor = StabilityEarlyStop()
    warmup_end = {lang: _validation_metric(10.0, 0.50) for lang in ("de", "en")}
    first_post_warmup = {lang: _validation_metric(10.0, 0.499) for lang in ("de", "en")}

    stable, _ = monitor.update(
        warmup_end, target_l0=10.0, cfg=cfg, eligible=False
    )
    assert not stable and monitor.stable_intervals == 0
    stable, _ = monitor.update(
        first_post_warmup, target_l0=10.0, cfg=cfg, eligible=True
    )
    assert stable and monitor.stable_intervals == 1


def _validation_metric(l0: float, fvu: float):
    return SimpleNamespace(l0=l0, fvu=fvu)
