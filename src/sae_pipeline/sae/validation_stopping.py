"""Versioned validation-objective stopping and independent atomic best exports."""

from dataclasses import asdict
import json
import math
import os
from pathlib import Path
from sae_pipeline.sae.checkpoint import write_raw_export
from sae_pipeline.sae.objective import inverse_simpson_warmup_steps


def eligibility_step(cfg, rule):
    return max(
        cfg.warmup_steps,
        cfg.l0_warmup_steps,
        cfg.l0_target_warmup_steps,
        inverse_simpson_warmup_steps(cfg)
        if cfg.inverse_simpson_floor > 0 and cfg.inverse_simpson_max_scale > 0
        else 0,
        rule.minimum_steps if rule.enabled else 0,
    )


def configure_stopping(site, rule):
    policy = dict(
        version=1,
        score=rule.score,
        eligible_after_step=eligibility_step(site.cfg, rule),
    )
    previous = site.stopping_state.get("policy")
    if previous != policy:
        if rule.score == "full_training_loss" or previous is not None:
            site.early_stop.best = None
            site.early_stop.checks_since_improve = 0
            site.stopping_state = dict(
                previous_policy=previous or {"score": "reconstruction", "version": 0},
                migration_step=site.step,
                baseline_initialized=False,
            )
            site.stopped = False
        site.stopping_state["policy"] = policy
    if rule.min_lr_scale == 1.0:
        site.early_stop.lr_scale = 1.0


def observe_full_best(site, evaluation):
    score = evaluation.stopping_score("full_training_loss")
    if not math.isfinite(score):
        raise FloatingPointError("Nonfinite full validation objective")
    previous = site.stopping_state.get("best_full_objective")
    if previous is not None and score >= previous["score"]:
        return
    folder = site.out_dir / "best_full_objective"
    folder.mkdir(exist_ok=True)
    destination = folder / f"best_step_{site.step:07d}.safetensors"
    temporary = destination.with_suffix(".partial.safetensors")
    write_raw_export(site.sae, temporary, input_scale=site.normalization_scale)
    os.replace(temporary, destination)
    record = dict(
        score=score,
        step=site.step,
        tokens=site.tokens_seen(),
        export=str(destination.relative_to(site.out_dir)).replace("\\", "/"),
        per_language=evaluation.training_objective,
        components=evaluation.loss_components,
        coefficients=evaluation.coefficients,
        metrics={lang: asdict(value) for lang, value in evaluation.metrics.items()},
    )
    site.stopping_state["best_full_objective"] = record
    temporary = folder / "best.json.new"
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    os.replace(temporary, folder / "best.json")


def prune_full_best(site):
    best = site.stopping_state.get("best_full_objective")
    if best is not None:
        for path in (site.out_dir / "best_full_objective").glob(
            "best_step_*.safetensors"
        ):
            if path.name != Path(best["export"]).name:
                path.unlink()
