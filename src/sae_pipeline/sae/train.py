"""SAE training on an activation cache: one job per (site, width, L0) on one device."""

from __future__ import annotations

import functools
import json
import logging
import math
import random
import shutil
import time
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch

from sae_pipeline.cache.reader import ActivationBuffer, ShardReader
from sae_pipeline.config import EarlyStoppingCfg, SAECfg
from sae_pipeline.eval.metrics import (
    ReconMetrics,
    streaming_reconstruction_metrics,
)
from sae_pipeline.eval.plots import (
    plot_training_curves,
    plot_validation_curves,
    refresh_plot,
)
from sae_pipeline.sae.checkpoint import (capture_random_state, load_sae_weights, load_trainer_state, restore_random_state, save_checkpoint_pair, write_raw_export)
from sae_pipeline.sae.checkpoint_source import (resolve_checkpoint)
from sae_pipeline.sae.early_stopping import StabilityEarlyStop
from sae_pipeline.sae.jumprelu import JumpReLUSAE
from sae_pipeline.sae.normalization import (
    centering_path,
    normalization_path,
    resolve_activation_center,
    resolve_whole_vector_normalization,
    write_normalization,
)
from sae_pipeline.sae.objective import (
    coefficients_at,
    inverse_simpson_warmup_steps,
    optimizer_step,
    scheduled_l0_target,
    training_loss,
)
from sae_pipeline.sae.resets import (
    active_subspace,
    load_reset_calibration,
    resample_underused_features,
    reset_l0_allowance,
    residual_reset_due,
)
from sae_pipeline.sae.step_log import record_firing, step_log
from sae_pipeline.sae.validation import prepare_validation_probes, validation_batches

log = logging.getLogger(__name__)


def build_sae_and_optimizer(
    cfg: SAECfg, *, d_in: int, d_sae: int, device: str
) -> tuple[JumpReLUSAE, torch.optim.Optimizer]:
    """Build the SAE and its Adam optimizer from one SAECfg, ready to train."""
    sae = JumpReLUSAE(
        d_in=d_in,
        d_sae=d_sae,
        bandwidth=cfg.bandwidth,
        pre_encoder_bias=(cfg.pre_encoder_bias or cfg.input_normalization == "whole_vector"),
    ).to(device)
    optim = torch.optim.Adam(
        sae.parameters(),
        lr=cfg.lr,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        eps=cfg.adam_eps,
    )
    return sae, optim


def seed_training(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _restore_resume_state(
    source_checkpoint,
    *,
    optim: torch.optim.Optimizer,
    ever_fired: torch.Tensor,
    feature_frequency: torch.Tensor,
    early_stop_monitor: "StabilityEarlyStop",
    buffer: ActivationBuffer,
    validation_readers: dict[str, ShardReader],
    validation_cache_signature: dict[str, dict[str, object]],
    early_stopping: EarlyStoppingCfg,
    n_steps: int,
    device: str,
) -> tuple[int, float | None, int, int | None]:
    """Restore optimizer, firing stats, early-stop state, buffer and RNG from a resume checkpoint.

    Mutates ``optim``, ``ever_fired``, ``feature_frequency``, ``early_stop_monitor``
    and ``buffer`` in place, and restores the global RNG state as a side effect.
    Returns (start_step, frequency_penalty_scale, feature_resets, reset_target_step).
    """
    assert source_checkpoint.trainer_state_path is not None
    trainer_state = load_trainer_state(source_checkpoint.trainer_state_path)
    state_step = int(trainer_state.get("global_step", -1))
    if state_step != source_checkpoint.step:
        raise ValueError(
            "Checkpoint pair has inconsistent steps: weights filename says "
            f"{source_checkpoint.step}, trainer state says {state_step}"
        )
    optim.load_state_dict(trainer_state["optimizer"])
    saved_fired = trainer_state["ever_fired"]
    if not isinstance(saved_fired, torch.Tensor) or saved_fired.shape != ever_fired.shape:
        raise ValueError(
            "Dead-feature state dimension mismatch: checkpoint has "
            f"{getattr(saved_fired, 'shape', None)}, current SAE has {ever_fired.shape}"
        )
    ever_fired.copy_(saved_fired.to(device=device, dtype=torch.bool))
    saved_frequency = trainer_state.get("feature_frequency")
    if saved_frequency is not None:
        if not isinstance(saved_frequency, torch.Tensor) or saved_frequency.shape != feature_frequency.shape:
            raise ValueError(
                "Feature-frequency state dimension mismatch: checkpoint has "
                f"{getattr(saved_frequency, 'shape', None)}, current SAE has {feature_frequency.shape}"
            )
        feature_frequency.copy_(saved_frequency.to(device=device, dtype=torch.float32))
    saved_scale = trainer_state.get("frequency_penalty_scale")
    frequency_penalty_scale = None if saved_scale is None else float(saved_scale)
    feature_resets = int(trainer_state.get("feature_resets", 0))
    saved_reset_target_step = trainer_state.get("reset_target_step")
    reset_target_step = (
        None if saved_reset_target_step is None else int(saved_reset_target_step)
    )
    saved_early_stop = trainer_state.get("validation_early_stop")
    saved_probe_signature = trainer_state.get("validation_cache_signature")
    if validation_readers and saved_probe_signature is not None:
        if saved_probe_signature != validation_cache_signature:
            raise ValueError(
                "Validation probe caches differ from the resume checkpoint"
            )
    elif validation_readers:
        log.warning(
            "Resume checkpoint has no validation-cache signature; "
            "early-stopping patience restarts"
        )
    if (
        isinstance(saved_early_stop, dict)
        and (not validation_readers or saved_probe_signature == validation_cache_signature)
    ):
        early_stop_monitor.load_state_dict(saved_early_stop)
    elif early_stopping.enabled and saved_probe_signature is not None:
        log.warning(
            "Resume checkpoint has no early-stopping state; stability patience restarts"
        )
    buffer.load_state_dict(trainer_state["activation_buffer"])
    restore_random_state(trainer_state["random_state"])
    start_step = state_step
    if start_step > n_steps:
        raise ValueError(
            f"Resume checkpoint is at step {start_step}, beyond n_steps target "
            f"{n_steps}"
        )
    log.info(
        "Resuming training from %s at global step %d (target %d)",
        source_checkpoint.weights_path,
        start_step,
        n_steps,
    )
    return start_step, frequency_penalty_scale, feature_resets, reset_target_step


def _reject_streaming_only_options(cfg: SAECfg, early_stopping: EarlyStoppingCfg) -> None:
    if early_stopping.enabled and early_stopping.mode != "stability":
        raise ValueError("Cached training supports early_stopping.mode='stability'; use streaming training for objective mode")
    if cfg.inverse_simpson_window_steps != 1:
        raise NotImplementedError("inverse_simpson_window_steps applies to streaming training only")
    if cfg.auxk_scale > 0:
        raise NotImplementedError("auxk_scale applies to streaming training only")


def _input_scale(cfg: SAECfg, cache_dir: str | Path, metadata_dir: Path, out_dir: Path, device: str) -> float:
    """The whole-vector normalisation scale (1 without normalisation), recorded next to the checkpoints."""
    if cfg.input_normalization != "whole_vector":
        return 1.0
    normalization = resolve_whole_vector_normalization(cache_dir, metadata_dir, device=device)
    if normalization_path(out_dir) != normalization_path(metadata_dir):
        write_normalization(normalization_path(out_dir), normalization)
    log.info(
        "Whole-vector input normalisation: c=%.8g, E[||x||²]=%.8g over %d tokens",
        normalization.input_scale, normalization.mean_squared_norm, normalization.n_tokens,
    )
    return normalization.input_scale


def _load_source_weights(cfg: SAECfg, sae: JumpReLUSAE, input_scale: float):
    """Load ``cfg.checkpoint_source`` into ``sae``, unfolding a raw export; None without a source."""
    if cfg.checkpoint_source is None:
        return None
    resume = cfg.checkpoint_mode == "resume"
    source = resolve_checkpoint(
        cfg.checkpoint_source,
        filename=cfg.checkpoint_filename,
        revision=cfg.checkpoint_revision,
        require_trainer_state=resume,
    )
    load_sae_weights(sae, source.weights_path)
    if cfg.checkpoint_format == "raw_export":
        if resume:
            raise ValueError("raw_export checkpoints can only be fine-tuned, not resumed")
        if cfg.input_normalization != "whole_vector":
            raise ValueError("raw_export fine-tuning requires whole_vector input normalisation")
        sae.unfold_raw_export(input_scale)
    return source


def _activation_center(
    cfg: SAECfg, cache_dir: str | Path, metadata_dir: Path, out_dir: Path, *, input_scale: float, d_in: int, device: str
) -> torch.Tensor | None:
    """The mean activation subtracted from every input under ``activation_centering: mean``."""
    if cfg.activation_centering != "mean":
        return None
    centre_metadata_dir = Path(cfg.centering_dir) if cfg.centering_dir else metadata_dir
    centre = resolve_activation_center(
        cache_dir, centre_metadata_dir, sample_tokens=cfg.centering_sample_tokens, input_scale=input_scale
    )
    center = torch.tensor(centre.mean, device=device, dtype=torch.float32)
    if center.numel() != d_in:
        raise ValueError("Activation centre has the wrong dimension")
    if centering_path(out_dir) != centering_path(centre_metadata_dir):
        shutil.copyfile(centering_path(centre_metadata_dir), centering_path(out_dir))
    return center


def _frequency_penalty(
    cfg: SAECfg, step: int, f: torch.Tensor, feature_frequency: torch.Tensor,
    reconstruction: torch.Tensor, scale: float | None,
) -> tuple[torch.Tensor, float | None]:
    """Penalty on latents firing above ``feature_frequency_threshold``, and its scale.

    The scale is fixed on first use so the penalty starts as a set fraction of
    the reconstruction loss.
    """
    if (
        cfg.feature_use_strategy not in {"frequency", "frequency_residual_reset"}
        or step < cfg.feature_frequency_start_step
    ):
        return reconstruction.new_zeros(()), scale
    weights = (
        (feature_frequency - cfg.feature_frequency_threshold)
        / max(1.0 - cfg.feature_frequency_threshold, 1.0e-6)
    ).clamp(0.0, 1.0).detach()
    weighted_activation = (weights * f).mean()
    if scale is None:
        target = cfg.feature_frequency_penalty_fraction * cfg.feature_frequency_penalty_multiplier * reconstruction.detach()
        scale = float((target / weighted_activation.detach().clamp_min(1.0e-8)).item())
    return scale * weighted_activation, scale


def _probe_metrics(
    sae: JumpReLUSAE,
    readers: dict[str, ShardReader],
    batches: Callable[[ShardReader], Iterable[torch.Tensor]],
    *,
    probe_tokens: int,
    step: int,
    out_dir: Path,
) -> dict[str, ReconMetrics]:
    """Metrics on every probe; per-token arrays go to ``eval_checkpoints/``."""
    eval_dir = out_dir / "eval_checkpoints"
    eval_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, ReconMetrics] = {}
    for split, reader in readers.items():
        metrics, arrays = streaming_reconstruction_metrics(
            sae, batches(reader), dead_threshold_tokens=probe_tokens, return_arrays=True
        )
        if not all(math.isfinite(float(value)) for value in (metrics.l0, metrics.fvu, metrics.dead_pct)):
            raise FloatingPointError(f"Non-finite evaluation metric at step {step} for {split}: {metrics}")
        np.savez_compressed(
            eval_dir / f"eval_arrays_step_{step:07d}_{split}.npz",
            firing_frequency=arrays.firing_frequency,
            l0_per_token=arrays.l0_per_token,
            recon_err_per_token=arrays.recon_err_per_token,
        )
        log.info(
            "eval step=%d split=%s fvu=%.6g l0=%.4f inactive@%d=%.2f%% D_IS=%.1f u_IS=%.2f%% D50=%.2f%% D90=%.2f%%",
            step, split, metrics.fvu, metrics.l0, metrics.n_tokens, metrics.dead_pct,
            metrics.d_is, metrics.u_is * 100.0, metrics.d_50, metrics.d_90,
        )
        results[split] = metrics
    sae.train()
    return results


def _validation_rows(
    metrics: dict[str, ReconMetrics], *, step: int, target_l0: float, stable: bool, l0_ok: bool,
    stable_intervals: int, stop_now: bool,
) -> list[dict[str, object]]:
    rows = []
    for split, split_metrics in metrics.items():
        row = asdict(split_metrics)
        row["inactive_pct"] = row.pop("dead_pct")
        row.update(
            step=step,
            language=split,
            target_l0=float(target_l0),
            probe_sample_tokens=split_metrics.n_tokens,
            inactive_sample_tokens=split_metrics.n_tokens,
            stable_interval=stable,
            l0_within_tolerance=l0_ok,
            stable_intervals=stable_intervals,
            early_stop_triggered=stop_now,
        )
        rows.append(row)
    return rows


def train_sae(
    cfg: SAECfg,
    cache_dir: str | Path,
    d_in: int,
    arch: str,
    d_sae: int,
    l0_target: int,
    out_dir: str | Path,
    device: str | None = None,
    normalization_dir: str | Path | None = None,
    validation_cache_dirs: dict[str, str | Path] | None = None,
    validation_tokens: int | None = None,
    validation_dead_tokens: int = 50_000,
    early_stopping: EarlyStoppingCfg | None = None,
) -> Path:
    """Train one SAE on an activation cache and return its last checkpoint."""
    early_stopping = early_stopping or EarlyStoppingCfg()
    _reject_streaming_only_options(cfg, early_stopping)
    seed_training(cfg.seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    validation_readers, validation_cache_signature, train_probe_reader, probe_tokens = prepare_validation_probes(
        cache_dir,
        d_in,
        validation_cache_dirs=validation_cache_dirs or {},
        validation_tokens=validation_tokens,
        validation_dead_tokens=validation_dead_tokens,
        early_stopping=early_stopping,
    )
    early_stop_monitor = StabilityEarlyStop()
    eval_interval = cfg.eval_interval_steps or early_stopping.evaluation_interval_steps
    log.info(
        "Training SAE: arch=%s d_in=%d d_sae=%d l0=%d steps=%d device=%s",
        arch, d_in, d_sae, l0_target, cfg.n_steps, device,
    )
    if early_stopping.enabled:
        log.info(
            "Validation early stopping: evaluate every %d steps after step %d; patience=%d; "
            "FVU relative tolerance=%.3g; EV absolute tolerance=%.3g; L0 tolerance=%.1f%%",
            eval_interval, early_stopping.minimum_steps, early_stopping.patience,
            early_stopping.fvu_relative_tolerance, early_stopping.ev_absolute_tolerance,
            100.0 * early_stopping.l0_relative_tolerance,
        )
    if cfg.inverse_simpson_floor > 0 and cfg.inverse_simpson_max_scale > 0:
        log.info(
            "Inverse-Simpson regularizer: utilization floor=%.4f, max scale=%.4g, warm-up=%d steps",
            cfg.inverse_simpson_floor, cfg.inverse_simpson_max_scale, inverse_simpson_warmup_steps(cfg),
        )

    metadata_dir = Path(normalization_dir) if normalization_dir else out_dir
    normalization_scale = _input_scale(cfg, cache_dir, metadata_dir, out_dir, device)
    sae, optim = build_sae_and_optimizer(cfg, d_in=d_in, d_sae=d_sae, device=device)
    buffer = ActivationBuffer(cache_dir=cache_dir, batch_size=cfg.batch_size, n_batches_in_buffer=cfg.n_batches_in_buffer)
    ever_fired = torch.zeros(d_sae, dtype=torch.bool, device=device)
    feature_frequency = torch.zeros(d_sae, dtype=torch.float32, device=device)
    frequency_penalty_scale: float | None = None
    feature_resets = 0
    reset_target_step: int | None = None

    start_step = 0
    source_checkpoint = _load_source_weights(cfg, sae, normalization_scale)
    resumed = source_checkpoint is not None and cfg.checkpoint_mode == "resume"
    if resumed:
        start_step, frequency_penalty_scale, feature_resets, reset_target_step = _restore_resume_state(
            source_checkpoint,
            optim=optim,
            ever_fired=ever_fired,
            feature_frequency=feature_frequency,
            early_stop_monitor=early_stop_monitor,
            buffer=buffer,
            validation_readers=validation_readers,
            validation_cache_signature=validation_cache_signature,
            early_stopping=early_stopping,
            n_steps=cfg.n_steps,
            device=device,
        )
    elif source_checkpoint is not None:
        log.info("Fine-tuning weights from %s with fresh optimizer and step numbering", source_checkpoint.weights_path)

    activation_center = _activation_center(
        cfg, cache_dir, metadata_dir, out_dir, input_scale=normalization_scale, d_in=d_in, device=device
    )
    if activation_center is not None and not resumed:
        # A resumed checkpoint is already centred.
        with torch.no_grad():
            sae.b_dec.sub_(activation_center)

    subspace = None
    if cfg.active_subspace_rank is not None:
        if cfg.active_subspace_rank > d_in:
            raise ValueError("active_subspace_rank cannot exceed activation dimension")
        subspace = active_subspace(
            cache_dir,
            sample_tokens=cfg.active_subspace_sample_tokens,
            rank=cfg.active_subspace_rank,
            input_scale=normalization_scale,
            centre=(activation_center if activation_center is not None else torch.zeros(d_in, device=device)),
            device=device,
            metadata_dir=metadata_dir,
        )

    log_path = out_dir / "train_log.jsonl"
    validation_log_path = out_dir / "validation_log.jsonl"
    title = f"{arch} d_sae={d_sae} L0*={l0_target}"
    probe_batches = functools.partial(
        validation_batches,
        n_tokens=probe_tokens,
        batch_size=cfg.batch_size,
        input_scale=normalization_scale,
        input_center=activation_center,
        device=device,
    )

    def write_checkpoint(step: int) -> Path:
        saved = save_checkpoint_pair(
            out_dir=out_dir,
            step=step,
            model_state=sae.state_dict(),
            trainer_state={
                "version": 1,
                "global_step": step,
                "optimizer": optim.state_dict(),
                "random_state": capture_random_state(),
                "ever_fired": ever_fired.detach().cpu(),
                "feature_frequency": feature_frequency.detach().cpu(),
                "frequency_penalty_scale": frequency_penalty_scale,
                "feature_resets": feature_resets,
                "reset_target_step": reset_target_step,
                "validation_early_stop": early_stop_monitor.state_dict(),
                "validation_cache_signature": validation_cache_signature,
                "activation_buffer": buffer.state_dict(),
            },
            keep_last=cfg.keep_last_checkpoints,
        )
        log.info("Wrote checkpoint pair %s and %s", saved.weights_path, saved.trainer_state_path)
        # The JSONL is flushed at every checkpoint step, so this plot remains
        # useful if a later allocation ends before the final checkpoint.
        refresh_plot(
            plot_training_curves,
            jsonl_path=log_path,
            out_dir=out_dir / "plots",
            target_l0=l0_target,
            title_prefix=title,
            description=f"training plot at step {step}",
        )
        return saved.weights_path

    t_start = time.time()
    final_checkpoint = source_checkpoint.weights_path if source_checkpoint is not None else None
    completed_step = start_step
    if resumed and start_step == cfg.n_steps:
        log.info("Resume checkpoint is already at target step %d; writing it into %s", cfg.n_steps, out_dir)
        final_checkpoint = write_checkpoint(start_step)
    log_mode = "a" if resumed else "w"
    with ExitStack() as stack:
        log_f = stack.enter_context(open(log_path, log_mode))
        validation_f = stack.enter_context(open(validation_log_path, log_mode)) if validation_readers else None
        for step in range(start_step + 1, cfg.n_steps + 1):
            completed_step = step
            x = buffer.next_batch().to(device, dtype=torch.float32)
            if normalization_scale != 1.0:
                x = x * normalization_scale
            if activation_center is not None:
                x = x - activation_center

            coefficients = coefficients_at(cfg, step, n_steps=cfg.n_steps)
            target_l0 = scheduled_l0_target(step, l0_target, cfg) + reset_l0_allowance(cfg, step, reset_target_step)
            terms = training_loss(sae, x, cfg, coefficients, target_l0=target_l0)
            record_firing(
                terms.f,
                feature_frequency=feature_frequency,
                ever_fired=ever_fired,
                ema_decay=cfg.feature_frequency_ema_decay,
            )
            feature_use_penalty, frequency_penalty_scale = _frequency_penalty(
                cfg, step, terms.f, feature_frequency, terms.reconstruction, frequency_penalty_scale
            )
            loss = terms.total + feature_use_penalty
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite training loss at step {step}: "
                    f"reconstruction={terms.reconstruction.detach().item()} "
                    f"l0_penalty={terms.l0_penalty.detach().item()}"
                )
            gradient_stats = optimizer_step(sae, optim, loss, x, cfg, step=step, lr=coefficients.lr)

            if residual_reset_due(cfg, step):
                calibration_x = calibration_residuals = None
                if cfg.residual_reset_calibration_tokens > 0:
                    calibration_x, calibration_residuals, calibrated_frequency = load_reset_calibration(
                        sae,
                        cache_dir,
                        sample_tokens=cfg.residual_reset_calibration_tokens,
                        pool_size=cfg.residual_reset_pool_size,
                        normalization_scale=normalization_scale,
                        activation_center=activation_center,
                        batch_size=cfg.batch_size,
                        device=device,
                    )
                    feature_frequency.copy_(calibrated_frequency.to(dtype=torch.float32))
                feature_resets += resample_underused_features(
                    sae, optim, x, terms.x_hat.detach(), feature_frequency, ever_fired, cfg,
                    subspace, calibration_x, calibration_residuals,
                )
                reset_target_step = step
                del calibration_x, calibration_residuals

            if step % cfg.log_every == 0:
                entry = step_log(
                    sae,
                    x,
                    step=step,
                    loss=loss,
                    terms=terms,
                    gradient_stats=gradient_stats,
                    lr=coefficients.lr,
                    lambda_l0=coefficients.lambda_l0,
                    target_l0=target_l0,
                    ever_fired=ever_fired,
                    feature_frequency=feature_frequency,
                    feature_use_penalty=float(feature_use_penalty.detach()),
                    feature_resets=feature_resets,
                )
                log_f.write(json.dumps(asdict(entry)) + "\n")
                log_f.flush()
                log.info(
                    "step=%d loss=%.4f recon=%.4f l0=%.1f dead=%.1f%% D_is=%.1f u_is=%.2f%% is_pen=%.4f lr=%.2e lam=%.2e",
                    step, entry.loss, entry.reconstruction_loss, entry.hard_l0, entry.dead_pct, entry.d_is,
                    entry.u_is * 100.0, entry.inverse_simpson_penalty, coefficients.lr, coefficients.lambda_l0,
                )

            validation_due = bool(validation_readers) and (step % eval_interval == 0 or step == cfg.n_steps)
            stop_now = False
            if validation_due:
                metrics = _probe_metrics(
                    sae, {"train_probe": train_probe_reader, **validation_readers}, probe_batches,
                    probe_tokens=probe_tokens, step=step, out_dir=out_dir,
                )
                stable, l0_ok = False, True
                if early_stopping.enabled:
                    stable, l0_ok = early_stop_monitor.update(
                        {split: metrics[split] for split in validation_readers},
                        target_l0=float(target_l0),
                        cfg=early_stopping,
                        eligible=step >= early_stopping.minimum_steps,
                    )
                    stop_now = (
                        step >= early_stopping.minimum_steps
                        and early_stop_monitor.stable_intervals >= early_stopping.patience
                    )
                for row in _validation_rows(
                    metrics, step=step, target_l0=target_l0, stable=stable, l0_ok=l0_ok,
                    stable_intervals=early_stop_monitor.stable_intervals, stop_now=stop_now,
                ):
                    validation_f.write(json.dumps(row) + "\n")
                validation_f.flush()
                refresh_plot(
                    plot_validation_curves,
                    validation_log_path,
                    out_dir / "plots",
                    title_prefix=title,
                    description=f"validation plot at step {step}",
                )
            if step % cfg.ckpt_every == 0 or step == cfg.n_steps or validation_due:
                # Save after validation so resume restores the latest patience state.
                final_checkpoint = write_checkpoint(step)
            if stop_now:
                log.info(
                    "Early stopping at step %d: validation FVU plateaued for %d "
                    "evaluations and L0 stayed within %.1f%% of target",
                    step, early_stop_monitor.stable_intervals, 100.0 * early_stopping.l0_relative_tolerance,
                )
                break

    log.info("Training done in %.1fs", time.time() - t_start)
    refresh_plot(
        plot_training_curves,
        jsonl_path=log_path,
        out_dir=out_dir / "plots",
        target_l0=l0_target,
        title_prefix=title,
        description="training plots",
    )
    if final_checkpoint is None:
        raise ValueError("Training produced no checkpoint; n_steps must be positive")
    if cfg.input_normalization == "whole_vector":
        export_path = write_raw_export(
            sae,
            out_dir / f"sae_export_step_{completed_step:07d}.safetensors",
            input_scale=normalization_scale,
            activation_center=activation_center,
        )
        log.info("Wrote raw-activation SAE export %s", export_path)
    return final_checkpoint
