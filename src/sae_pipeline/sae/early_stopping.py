"""Stopping rules on held-out validation: metric stability (cached) and objective plateau (streaming)."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from sae_pipeline.config import EarlyStoppingCfg
from sae_pipeline.eval.metrics import ReconMetrics


@dataclass
class StabilityEarlyStop:
    previous: dict[str, ReconMetrics] | None = None
    stable_intervals: int = 0

    def state_dict(self) -> dict[str, object]:
        return {
            "previous": (
                None
                if self.previous is None
                else {language: asdict(metrics) for language, metrics in self.previous.items()}
            ),
            "stable_intervals": self.stable_intervals,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        previous = state.get("previous")
        self.previous = (
            None
            if previous is None
            else {
                str(language): ReconMetrics(**metrics)
                for language, metrics in dict(previous).items()
            }
        )
        self.stable_intervals = int(state.get("stable_intervals", 0))
        if self.stable_intervals < 0:
            raise ValueError("Validation early-stopping state has a negative interval count")

    def update(
        self,
        current: dict[str, ReconMetrics],
        *,
        target_l0: float,
        cfg: EarlyStoppingCfg,
        eligible: bool = True,
    ) -> tuple[bool, bool]:
        """Return (metrics_stable, target_l0_reached) for this evaluation."""
        l0_ok = all(
            abs(metrics.l0 - target_l0)
            <= cfg.l0_relative_tolerance * max(abs(target_l0), 1.0)
            for metrics in current.values()
        )
        if not eligible:
            self.previous = current
            self.stable_intervals = 0
            return False, l0_ok
        if self.previous is None or set(self.previous) != set(current):
            stable = False
        else:
            previous_l0_ok = all(
                abs(metrics.l0 - target_l0)
                <= cfg.l0_relative_tolerance * max(abs(target_l0), 1.0)
                for metrics in self.previous.values()
            )
            stable = l0_ok and previous_l0_ok and all(
                abs(current[language].fvu - self.previous[language].fvu)
                <= min(
                    cfg.ev_absolute_tolerance,
                    cfg.fvu_relative_tolerance
                    * max(abs(self.previous[language].fvu), 1.0e-12),
                )
                for language in current
            )
        self.stable_intervals = self.stable_intervals + 1 if stable else 0
        self.previous = current
        return stable, l0_ok


@dataclass
class ObjectiveEarlyStop:
    """Plateau-driven LR reduction and stopping on a held-out objective.

    The objective (lower is better) is the training loss evaluated on
    validation data. An evaluation is a new best when it is lower than the
    best so far by more than ``improvement_rel_tol``. Once ``eligible`` (e.g.
    after warmup), every non-improving evaluation counts towards a plateau:
    every ``lr_plateau_patience`` of them scale the learning rate by
    ``lr_reduce_factor`` (floored at ``min_lr_scale``), and ``patience`` of
    them stop the run.
    """

    best: float | None = None
    checks_since_improve: int = 0
    lr_scale: float = 1.0

    def state_dict(self) -> dict[str, object]:
        return {
            "best": self.best,
            "checks_since_improve": self.checks_since_improve,
            "lr_scale": self.lr_scale,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.best = None if state.get("best") is None else float(state["best"])  # type: ignore[arg-type]
        self.checks_since_improve = int(state.get("checks_since_improve", 0))  # type: ignore[arg-type]
        self.lr_scale = float(state.get("lr_scale", 1.0))  # type: ignore[arg-type]
        if self.checks_since_improve < 0:
            raise ValueError("Objective early-stopping state has a negative check count")
        if not 0.0 < self.lr_scale <= 1.0:
            raise ValueError("Objective early-stopping lr_scale must be in (0, 1]")

    def update(
        self, objective: float, cfg: EarlyStoppingCfg, *, eligible: bool = True
    ) -> tuple[bool, bool, bool]:
        """Return ``(is_best, lr_reduced, should_stop)`` for this evaluation."""
        if not math.isfinite(objective):
            raise FloatingPointError(f"Non-finite validation objective: {objective}")
        if self.best is None or objective < self.best * (1.0 - cfg.improvement_rel_tol):
            self.best = objective
            self.checks_since_improve = 0
            return True, False, False
        if not eligible:
            return False, False, False
        self.checks_since_improve += 1
        lr_reduced = False
        if (
            self.checks_since_improve < cfg.patience
            and self.checks_since_improve % cfg.lr_plateau_patience == 0
            and self.lr_scale > cfg.min_lr_scale
        ):
            self.lr_scale = max(cfg.min_lr_scale, self.lr_scale * cfg.lr_reduce_factor)
            lr_reduced = True
        return False, lr_reduced, self.checks_since_improve >= cfg.patience
