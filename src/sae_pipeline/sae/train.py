"""SAE training loop. Single-GPU; one job per (layer, component, arch, width, L0)."""

from __future__ import annotations

import json
import logging
import math
import random
import shutil
import time
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from sae_pipeline.cache.reader import ActivationBuffer, ShardReader
from sae_pipeline.config import EarlyStoppingCfg, SAECfg
from sae_pipeline.eval.metrics import (
    ReconMetrics,
    dead_pct_from_fired,
    firing_concentration,
    streaming_reconstruction_metrics,
)
from sae_pipeline.eval.plots import plot_training_curves, plot_validation_curves, refresh_plot
from sae_pipeline.sae.base import SparseAutoencoder
from sae_pipeline.sae.checkpoint import (
    capture_random_state,
    load_sae_weights,
    load_trainer_state,
    resolve_checkpoint,
    restore_random_state,
    save_checkpoint_pair,
    write_raw_export,
)
from sae_pipeline.sae.jumprelu import JumpReLUSAE
from sae_pipeline.sae.normalization import (
    centering_path,
    normalization_path,
    resolve_activation_center,
    resolve_whole_vector_normalization,
    write_normalization,
)
from sae_pipeline.sae.objective import (
    GradientStats,
    LossTerms,
    coefficients_at,
    inverse_simpson_warmup_steps,
    optimizer_step,
    training_loss,
)

log = logging.getLogger(__name__)


def build_sae(
    arch: str,
    d_in: int,
    d_sae: int,
    bandwidth: float = 0.001,
    pre_encoder_bias: bool = True,
) -> SparseAutoencoder:
    if arch == "jumprelu":
        return JumpReLUSAE(
            d_in=d_in,
            d_sae=d_sae,
            bandwidth=bandwidth,
            pre_encoder_bias=pre_encoder_bias,
        )
    raise NotImplementedError(f"SAE arch {arch!r} not yet implemented (registry stub).")


def build_sae_and_optimizer(
    cfg: SAECfg, *, arch: str, d_in: int, d_sae: int, device: str
) -> tuple[SparseAutoencoder, torch.optim.Optimizer]:
    """Build the SAE and its Adam optimizer from one SAECfg, ready to train."""
    sae = build_sae(
        arch,
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


@dataclass
class StepLog:
    step: int
    loss: float
    mse: float
    l0_penalty: float
    hard_l0: float
    lr: float
    lambda_l0: float
    dead_pct: float
    ste_window_pct: float
    theta_grad_norm: float
    encoder_grad_norm: float
    reconstruction_loss: float
    feature_use_penalty: float
    active_features: int
    top1_activation_mass_pct: float
    max_firing_frequency: float
    feature_resets: int
    target_l0: float
    d_is: float = 0.0
    u_is: float = 0.0
    inverse_simpson_penalty: float = 0.0



@torch.no_grad()
def record_firing(
    f: torch.Tensor,
    *,
    feature_frequency: torch.Tensor,
    ever_fired: torch.Tensor,
    ema_decay: float,
) -> None:
    """Fold one batch of latent activations into the firing statistics."""
    fired = f.detach() > 0
    feature_frequency.mul_(ema_decay).add_(fired.float().mean(dim=0), alpha=1.0 - ema_decay)
    ever_fired |= fired.any(dim=0)


@torch.no_grad()
def step_log(
    sae: SparseAutoencoder,
    x: torch.Tensor,
    *,
    step: int,
    loss: torch.Tensor,
    terms: LossTerms,
    gradient_stats: GradientStats,
    lr: float,
    lambda_l0: float,
    target_l0: float,
    ever_fired: torch.Tensor,
    feature_frequency: torch.Tensor,
    feature_use_penalty: float = 0.0,
    feature_resets: int = 0,
) -> StepLog:
    """Assemble one training-log row; costs a few device syncs, so call it sparingly."""
    concentration = firing_concentration(feature_frequency)
    return StepLog(
        step=step,
        loss=float(loss.detach()),
        mse=float(terms.mse.detach()),
        l0_penalty=float(terms.l0_penalty.detach()),
        hard_l0=sae.hard_l0(x).item(),  # type: ignore[attr-defined]
        lr=lr,
        lambda_l0=lambda_l0,
        dead_pct=dead_pct_from_fired(ever_fired),
        ste_window_pct=gradient_stats.ste_window_pct,
        theta_grad_norm=gradient_stats.theta_grad_norm,
        encoder_grad_norm=gradient_stats.encoder_grad_norm,
        reconstruction_loss=float(terms.reconstruction.detach()),
        feature_use_penalty=feature_use_penalty,
        active_features=concentration.active_features,
        top1_activation_mass_pct=concentration.top1_activation_mass_pct,
        max_firing_frequency=concentration.max_firing_frequency,
        feature_resets=feature_resets,
        target_l0=target_l0,
        d_is=terms.d_is,
        u_is=terms.u_is,
        inverse_simpson_penalty=float(terms.inverse_simpson_penalty.detach()),
    )


@dataclass
class _ValidationEarlyStop:
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


def _validation_batches(
    reader: ShardReader,
    *,
    n_tokens: int,
    batch_size: int,
    input_scale: float,
    input_center: torch.Tensor | None,
    device: str,
) -> Iterable[torch.Tensor]:
    """Yield the same bounded validation prefix deterministically each time."""
    if n_tokens <= 0 or n_tokens > reader.manifest.total_tokens:
        raise ValueError(
            f"Requested {n_tokens} validation vectors from a cache containing "
            f"{reader.manifest.total_tokens}"
        )
    seen = 0
    for shard in reader.iter_shards():
        for start in range(0, shard.shape[0], batch_size):
            if seen >= n_tokens:
                return
            take = min(batch_size, n_tokens - seen)
            # CPU float32 caches may alias the mapped shard after .to().
            batch = shard[start : start + take].to(device, dtype=torch.float32, copy=True)
            if input_scale != 1.0:
                batch.mul_(input_scale)
            if input_center is not None:
                batch.sub_(input_center)
            seen += batch.shape[0]
            yield batch
    if seen != n_tokens:
        raise ValueError(
            f"Validation cache yielded {seen} vectors, expected {n_tokens}"
        )


def _probe_cache_signature(readers: dict[str, ShardReader]) -> dict[str, dict[str, object]]:
    """Identify the ordered probe datasets used by validation trajectories."""
    return {
        language: {
            "run_id": reader.manifest.run_id,
            "model": reader.manifest.model,
            "layer": reader.manifest.layer,
            "component": reader.manifest.component,
            "partition": reader.manifest.partition,
            "language": reader.manifest.language,
            "d_activation": reader.manifest.d_activation,
            "total_tokens": reader.manifest.total_tokens,
            "shard_paths": list(reader.manifest.shard_paths),
            "dataset_fingerprint": reader.manifest.dataset_fingerprint,
        }
        for language, reader in sorted(readers.items())
    }


def _seed_training(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _scheduled_l0_target(step: int, base_target: int, cfg: SAECfg) -> float:
    if cfg.l0_target_start is None or cfg.l0_target_warmup_steps <= 0:
        return float(base_target)
    alpha = min(1.0, step / cfg.l0_target_warmup_steps)
    return float(cfg.l0_target_start + alpha * (base_target - cfg.l0_target_start))


def _active_subspace(
    cache_dir: str | Path,
    *,
    sample_tokens: int,
    rank: int,
    input_scale: float,
    centre: torch.Tensor,
    device: str,
    metadata_dir: Path,
) -> torch.Tensor:
    """Compute and cache leading centred activation directions."""
    from safetensors import safe_open

    path = metadata_dir / f"active_subspace_r{rank}.pt"
    if path.exists():
        basis = torch.load(path, map_location=device, weights_only=True)
        if tuple(basis.shape) == (centre.numel(), rank):
            return basis.to(device=device, dtype=torch.float32)

    manifest = ActivationBuffer(cache_dir, batch_size=1, n_batches_in_buffer=1)._reader.manifest
    chunks: list[torch.Tensor] = []
    seen = 0
    for name in manifest.shard_paths:
        if seen >= sample_tokens:
            break
        with safe_open(str(Path(cache_dir) / name), framework="pt", device="cpu") as handle:
            part = handle.get_tensor("x")[: sample_tokens - seen]
        chunks.append(part.to(device=device, dtype=torch.float32) * input_scale - centre)
        seen += part.shape[0]
    if not chunks:
        raise ValueError(f"Cannot compute an active subspace from {cache_dir}")
    sample = torch.cat(chunks, dim=0)
    _, _, basis = torch.pca_lowrank(sample, q=rank, center=False)
    basis = basis[:, :rank].contiguous()
    metadata_dir.mkdir(parents=True, exist_ok=True)
    torch.save(basis.detach().cpu(), path)
    return basis


@torch.no_grad()
def _load_reset_calibration(
    sae: JumpReLUSAE,
    cache_dir: str | Path,
    *,
    sample_tokens: int,
    pool_size: int,
    normalization_scale: float,
    activation_center: torch.Tensor | None,
    batch_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load a bounded calibration sample and measure exact reset statistics.

    The returned tensors are the normalised calibration activations, a pool of
    highest-error residuals, and per-latent firing frequencies.  Resetting from
    one optimisation batch was too noisy for L26; this pass makes threshold
    calibration depend on the requested support window instead.
    """
    reader = ShardReader(cache_dir)
    chunks: list[torch.Tensor] = []
    seen = 0
    for shard in reader.iter_shards():
        if seen >= sample_tokens:
            break
        take = min(shard.shape[0], sample_tokens - seen)
        chunks.append(shard[:take].contiguous())
        seen += take
    if seen == 0:
        raise ValueError(f"Cannot calibrate a reset from empty cache {cache_dir}")
    calibration_x = torch.cat(chunks, dim=0).to(device=device, dtype=torch.float32)
    if normalization_scale != 1.0:
        calibration_x.mul_(normalization_scale)
    if activation_center is not None:
        calibration_x.sub_(activation_center)

    frequencies = torch.zeros(sae.d_sae, dtype=torch.float64, device=device)
    best_errors = torch.empty(0, dtype=torch.float32, device=device)
    best_residuals = torch.empty((0, sae.d_in), dtype=torch.float32, device=device)
    for start in range(0, calibration_x.shape[0], batch_size):
        x = calibration_x[start : start + batch_size]
        x_hat, f = sae(x)
        frequencies.add_((f > 0).sum(dim=0, dtype=torch.float64))
        errors = (x - x_hat).square().sum(dim=-1)
        k = min(pool_size, errors.numel())
        values, indices = torch.topk(errors, k=k, largest=True, sorted=False)
        residuals = (x - x_hat)[indices].detach()
        if best_errors.numel():
            values = torch.cat([best_errors, values])
            residuals = torch.cat([best_residuals, residuals])
        keep = min(pool_size, values.numel())
        keep_values, keep_indices = torch.topk(values, k=keep, largest=True, sorted=False)
        best_errors = keep_values
        best_residuals = residuals[keep_indices]
    return calibration_x, best_residuals, frequencies / calibration_x.shape[0]


@torch.no_grad()
def _zero_optimizer_slices(
    optim: torch.optim.Optimizer,
    parameter: torch.Tensor,
    indices: torch.Tensor,
    *,
    column_parameter: bool,
) -> None:
    """Reset Adam moments for selected SAE columns after feature replacement."""

    state = optim.state.get(parameter)
    if not state:
        return
    for value in state.values():
        if not isinstance(value, torch.Tensor) or value.shape != parameter.shape:
            continue
        if column_parameter:
            value[:, indices] = 0
        else:
            value[indices] = 0


@torch.no_grad()
def _resample_underused_features(
    sae: JumpReLUSAE,
    optim: torch.optim.Optimizer,
    x: torch.Tensor,
    x_hat: torch.Tensor,
    feature_frequency: torch.Tensor,
    ever_fired: torch.Tensor,
    cfg: SAECfg,
    active_subspace: torch.Tensor | None = None,
    calibration_x: torch.Tensor | None = None,
    calibration_residuals: torch.Tensor | None = None,
) -> int:
    """Replace low-frequency features with high-error residual directions.

    This is intentionally a bounded intervention: it touches only a fixed
    number of columns and resets the matching Adam moments.  The starting SAE
    is a dictionary initialisation, so preserving the identities of replaced
    latents is not a requirement.
    """

    candidates = torch.nonzero(
        feature_frequency < cfg.residual_reset_frequency_threshold,
        as_tuple=False,
    ).flatten()
    if candidates.numel() == 0:
        return 0
    n_reset = min(cfg.residual_reset_max_features, int(candidates.numel()))
    candidates = candidates[torch.randperm(candidates.numel(), device=x.device)[:n_reset]]

    residual = x - x_hat
    if calibration_x is not None and calibration_residuals is not None:
        if calibration_residuals.shape[0] < n_reset:
            raise ValueError("Reset residual pool is smaller than the requested reset")
        directions = calibration_residuals[
            torch.randperm(calibration_residuals.shape[0], device=x.device)[:n_reset]
        ]
    else:
        errors = residual.square().sum(dim=-1)
        pool_size = min(cfg.residual_reset_pool_size, x.shape[0])
        probabilities = errors.clamp_min(torch.finfo(errors.dtype).eps)
        probabilities = probabilities / probabilities.sum()
        sample_indices = torch.multinomial(probabilities, pool_size, replacement=False)
        directions = residual[sample_indices[:n_reset]]
    norms = directions.norm(dim=-1, keepdim=True)
    invalid = norms.squeeze(-1) <= 1.0e-8
    if invalid.any():
        directions[invalid] = torch.randn_like(directions[invalid])
        norms = directions.norm(dim=-1, keepdim=True)
    directions = directions / norms.clamp_min(1.0e-8)
    if active_subspace is not None:
        directions = directions @ active_subspace @ active_subspace.T
        norms = directions.norm(dim=-1, keepdim=True)
        invalid = norms.squeeze(-1) <= 1.0e-8
        if invalid.any():
            replacement = torch.randn(
                int(invalid.sum().item()), active_subspace.shape[1], device=x.device
            ) @ active_subspace.T
            directions[invalid] = replacement
            norms = directions.norm(dim=-1, keepdim=True)
        directions = directions / norms.clamp_min(1.0e-8)

    live = feature_frequency > 0
    live_norms = sae.W_enc[:, live].norm(dim=0)
    encoder_scale = (
        0.2 * live_norms.median().item() if live_norms.numel() else 0.2
    )
    encoder_scale = max(encoder_scale, 1.0e-3)

    sae.W_dec[:, candidates] = directions.T
    sae.W_enc[:, candidates] = directions.T * encoder_scale
    sae.b_enc[candidates] = 0

    calibration = x if calibration_x is None else calibration_x
    centred = calibration - sae.b_dec if sae.pre_encoder_bias else calibration
    projection = torch.relu(centred @ sae.W_enc[:, candidates] + sae.b_enc[candidates])
    target_frequency = cfg.residual_reset_delta_l0 / max(1, n_reset)
    quantile = torch.quantile(projection, 1.0 - target_frequency, dim=0)
    threshold = quantile.clamp_min(max(cfg.bandwidth, 1.0e-6))
    sae.log_theta[candidates] = threshold.log()

    _zero_optimizer_slices(optim, sae.W_dec, candidates, column_parameter=True)
    _zero_optimizer_slices(optim, sae.W_enc, candidates, column_parameter=True)
    _zero_optimizer_slices(optim, sae.b_enc, candidates, column_parameter=False)
    _zero_optimizer_slices(optim, sae.log_theta, candidates, column_parameter=False)
    ever_fired[candidates] = False
    feature_frequency[candidates] = 0
    return n_reset


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
    _seed_training(cfg.seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    early_stopping = early_stopping or EarlyStoppingCfg()
    validation_cache_dirs = validation_cache_dirs or {}
    if early_stopping.enabled:
        if not validation_cache_dirs:
            raise ValueError("Validation-based early stopping requires validation caches")
        if validation_tokens is None or validation_tokens <= 0:
            raise ValueError("Validation-based early stopping requires a positive token count")
        if len(validation_cache_dirs) < 2:
            raise ValueError("Bilingual early stopping requires two language validation caches")
    validation_readers = {
        language: ShardReader(path)
        for language, path in sorted(validation_cache_dirs.items())
    }
    validation_cache_signature = _probe_cache_signature(validation_readers)
    train_probe_reader: ShardReader | None = None
    probe_tokens = validation_tokens if validation_tokens is not None else validation_dead_tokens
    if validation_readers:
        if probe_tokens <= 0:
            raise ValueError("Validation trajectories require a positive fixed-probe token count")
        undersized = {
            language: reader.manifest.total_tokens
            for language, reader in validation_readers.items()
            if reader.manifest.total_tokens < probe_tokens
        }
        if undersized:
            raise ValueError(
                f"Validation caches are smaller than the {probe_tokens}-token probe: {undersized}"
            )
        train_probe_reader = ShardReader(cache_dir)
        train_manifest = train_probe_reader.manifest
        if not train_manifest.complete or train_manifest.partition != "train":
            raise ValueError("Training probe must come from a complete training cache")
        if train_manifest.d_activation != d_in:
            raise ValueError("Training probe activation dimension differs from the SAE")
        if train_probe_reader.manifest.total_tokens < probe_tokens:
            raise ValueError(
                f"Training cache has {train_probe_reader.manifest.total_tokens} tokens; "
                f"the fixed probe requires {probe_tokens}"
            )
        for language, reader in validation_readers.items():
            manifest = reader.manifest
            if not manifest.complete or manifest.partition != "validation":
                raise ValueError(f"{language} probe must come from a complete validation cache")
            if manifest.language != language:
                raise ValueError(
                    f"Validation cache keyed as {language!r} declares language "
                    f"{manifest.language!r}"
                )
            if (manifest.model, manifest.layer, manifest.component) != (
                train_manifest.model,
                train_manifest.layer,
                train_manifest.component,
            ):
                raise ValueError(f"{language} validation cache belongs to a different site/model")
            if manifest.d_activation != d_in:
                raise ValueError(f"{language} validation activation dimension differs from the SAE")
        log.info(
            "Initialized fixed %d-token train probe from %s",
            probe_tokens,
            cache_dir,
        )
    early_stop_monitor = _ValidationEarlyStop()
    validation_log_path = out_dir / "validation_log.jsonl"
    eval_interval = cfg.eval_interval_steps or early_stopping.evaluation_interval_steps
    log.info(
        "Training SAE: arch=%s d_in=%d d_sae=%d l0=%d steps=%d device=%s",
        arch, d_in, d_sae, l0_target, cfg.n_steps, device,
    )
    if early_stopping.enabled:
        log.info(
            "Validation early stopping: evaluate every %d steps after step %d; patience=%d; "
            "FVU relative tolerance=%.3g; EV absolute tolerance=%.3g; L0 tolerance=%.1f%%",
            eval_interval,
            early_stopping.minimum_steps,
            early_stopping.patience,
            early_stopping.fvu_relative_tolerance,
            early_stopping.ev_absolute_tolerance,
            100.0 * early_stopping.l0_relative_tolerance,
        )
    if cfg.inverse_simpson_floor > 0 and cfg.inverse_simpson_max_scale > 0:
        log.info(
            "Inverse-Simpson regularizer: utilization floor=%.4f, max scale=%.4g, "
            "warm-up=%d steps",
            cfg.inverse_simpson_floor,
            cfg.inverse_simpson_max_scale,
            inverse_simpson_warmup_steps(cfg),
        )

    normalization_scale = 1.0
    metadata_dir = Path(normalization_dir) if normalization_dir else out_dir
    if cfg.input_normalization == "whole_vector":
        normalization = resolve_whole_vector_normalization(cache_dir, metadata_dir, device=device)
        if normalization_path(out_dir) != normalization_path(metadata_dir):
            write_normalization(normalization_path(out_dir), normalization)
        normalization_scale = normalization.input_scale
        log.info(
            "Whole-vector input normalisation: c=%.8g, E[||x||²]=%.8g over %d tokens",
            normalization.input_scale,
            normalization.mean_squared_norm,
            normalization.n_tokens,
        )

    sae, optim = build_sae_and_optimizer(cfg, arch=arch, d_in=d_in, d_sae=d_sae, device=device)

    buffer = ActivationBuffer(
        cache_dir=cache_dir,
        batch_size=cfg.batch_size,
        n_batches_in_buffer=cfg.n_batches_in_buffer,
    )

    # Track which latents have ever fired (for dead-feature monitoring).
    ever_fired = torch.zeros(d_sae, dtype=torch.bool, device=device)
    feature_frequency = torch.zeros(d_sae, dtype=torch.float32, device=device)
    frequency_penalty_scale: float | None = None
    feature_resets = 0
    reset_target_step: int | None = None

    start_step = 0
    source_checkpoint = None
    if cfg.checkpoint_source is not None:
        is_resume = cfg.checkpoint_mode == "resume"
        source_checkpoint = resolve_checkpoint(
            cfg.checkpoint_source,
            filename=cfg.checkpoint_filename,
            revision=cfg.checkpoint_revision,
            require_trainer_state=is_resume,
        )
        load_sae_weights(sae, source_checkpoint.weights_path)
        if cfg.checkpoint_format == "raw_export":
            if is_resume:
                raise ValueError("raw_export checkpoints can only be fine-tuned, not resumed")
            if not isinstance(sae, JumpReLUSAE):
                raise ValueError("raw_export conversion is currently implemented for JumpReLU only")
            if cfg.input_normalization != "whole_vector":
                raise ValueError("raw_export fine-tuning requires whole_vector input normalisation")
            sae.unfold_raw_export(normalization_scale)
        if is_resume:
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
            if start_step > cfg.n_steps:
                raise ValueError(
                    f"Resume checkpoint is at step {start_step}, beyond n_steps target "
                    f"{cfg.n_steps}"
                )
            log.info(
                "Resuming training from %s at global step %d (target %d)",
                source_checkpoint.weights_path,
                start_step,
                cfg.n_steps,
            )
        else:
            log.info(
                "Fine-tuning weights from %s with fresh optimizer and step numbering",
                source_checkpoint.weights_path,
            )

    activation_center: torch.Tensor | None = None
    if cfg.activation_centering == "mean":
        centre = resolve_activation_center(
            cache_dir,
            metadata_dir,
            sample_tokens=cfg.centering_sample_tokens,
            input_scale=normalization_scale,
        )
        activation_center = torch.tensor(centre.mean, device=device, dtype=torch.float32)
        if activation_center.numel() != d_in:
            raise ValueError("Activation centre has the wrong dimension")
        if cfg.checkpoint_mode != "resume" or source_checkpoint is None:
            with torch.no_grad():
                # A resumed canonical checkpoint is already centred.
                sae.b_dec.sub_(activation_center)
        if centering_path(out_dir) != centering_path(metadata_dir):
            shutil.copyfile(centering_path(metadata_dir), centering_path(out_dir))

    active_subspace = None
    if cfg.active_subspace_rank is not None:
        if cfg.active_subspace_rank > d_in:
            raise ValueError("active_subspace_rank cannot exceed activation dimension")
        active_subspace = _active_subspace(
            cache_dir,
            sample_tokens=cfg.active_subspace_sample_tokens,
            rank=cfg.active_subspace_rank,
            input_scale=normalization_scale,
            centre=(activation_center if activation_center is not None else torch.zeros(d_in, device=device)),
            device=device,
            metadata_dir=metadata_dir,
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
        log.info(
            "Wrote checkpoint pair %s and %s", saved.weights_path, saved.trainer_state_path
        )
        # The JSONL is flushed at every checkpoint step, so this plot remains
        # useful if a later allocation ends before the final checkpoint.
        refresh_plot(
            plot_training_curves,
            jsonl_path=log_path,
            out_dir=out_dir / "plots",
            target_l0=l0_target,
            title_prefix=f"{arch}  d_sae={d_sae}  L0*={l0_target}",
            description=f"training plot at step {step}",
        )
        return saved.weights_path

    log_path = out_dir / "train_log.jsonl"
    log_mode = "a" if cfg.checkpoint_source is not None and cfg.checkpoint_mode == "resume" else "w"
    validation_log_mode = log_mode

    t_start = time.time()
    final_checkpoint = source_checkpoint.weights_path if source_checkpoint is not None else None
    completed_step = start_step
    if source_checkpoint is not None and cfg.checkpoint_mode == "resume" and start_step == cfg.n_steps:
        log.info(
            "Resume checkpoint is already at target step %d; writing it into %s",
            cfg.n_steps, out_dir,
        )
        final_checkpoint = write_checkpoint(start_step)
    with ExitStack() as stack:
        log_f = stack.enter_context(open(log_path, log_mode))
        validation_f = (
            stack.enter_context(open(validation_log_path, validation_log_mode))
            if validation_readers
            else None
        )
        for step in range(start_step + 1, cfg.n_steps + 1):
            completed_step = step
            x = buffer.next_batch().to(device, dtype=torch.float32)
            if normalization_scale != 1.0:
                x = x * normalization_scale
            if activation_center is not None:
                x = x - activation_center

            coefficients = coefficients_at(cfg, step, n_steps=cfg.n_steps)
            lr = coefficients.lr
            lam = coefficients.lambda_l0

            target_l0 = _scheduled_l0_target(step, l0_target, cfg)
            if reset_target_step is not None:
                elapsed = step - reset_target_step
                if elapsed < cfg.residual_reset_anneal_steps:
                    target_l0 += cfg.residual_reset_delta_l0 * (
                        1.0 - elapsed / max(1, cfg.residual_reset_anneal_steps)
                    )
            terms = training_loss(sae, x, cfg, coefficients, target_l0=target_l0)
            x_hat, f = terms.x_hat, terms.f

            record_firing(
                f,
                feature_frequency=feature_frequency,
                ever_fired=ever_fired,
                ema_decay=cfg.feature_frequency_ema_decay,
            )

            feature_use_penalty = terms.reconstruction.new_zeros(())
            if (
                cfg.feature_use_strategy in {"frequency", "frequency_residual_reset"}
                and step >= cfg.feature_frequency_start_step
            ):
                frequency_weights = (
                    (feature_frequency - cfg.feature_frequency_threshold)
                    / max(1.0 - cfg.feature_frequency_threshold, 1.0e-6)
                ).clamp(0.0, 1.0).detach()
                weighted_activation = (frequency_weights * f).mean()
                if frequency_penalty_scale is None:
                    target_penalty = (
                        cfg.feature_frequency_penalty_fraction
                        * cfg.feature_frequency_penalty_multiplier
                        * terms.reconstruction.detach()
                    )
                    frequency_penalty_scale = float(
                        (target_penalty / weighted_activation.detach().clamp_min(1.0e-8)).item()
                    )
                feature_use_penalty = frequency_penalty_scale * weighted_activation

            loss = terms.total + feature_use_penalty
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite training loss at step {step}: "
                    f"reconstruction={terms.reconstruction.detach().item()} "
                    f"l0_penalty={terms.l0_penalty.detach().item()}"
                )

            gradient_stats = optimizer_step(sae, optim, loss, x, cfg, step=step, lr=lr)

            if (
                cfg.feature_use_strategy in {"residual_reset", "frequency_residual_reset"}
                and step >= cfg.residual_reset_start_step
                and (step - cfg.residual_reset_start_step) % cfg.residual_reset_every_steps == 0
            ):
                calibration_x = None
                calibration_residuals = None
                if cfg.residual_reset_calibration_tokens > 0:
                    calibration_x, calibration_residuals, calibrated_frequency = _load_reset_calibration(
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
                feature_resets += _resample_underused_features(
                    sae,
                    optim,
                    x,
                    x_hat.detach(),
                    feature_frequency,
                    ever_fired,
                    cfg,
                    active_subspace,
                    calibration_x,
                    calibration_residuals,
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
                    lr=lr,
                    lambda_l0=lam,
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
                    step, entry.loss, entry.reconstruction_loss, entry.hard_l0,
                    entry.dead_pct, entry.d_is, entry.u_is * 100.0, entry.inverse_simpson_penalty, lr, lam,
                )

            validation_due = bool(validation_readers) and (
                step % eval_interval == 0
                or step == cfg.n_steps
            )
            checkpoint_due = (
                step % cfg.ckpt_every == 0 or step == cfg.n_steps or validation_due
            )

            if validation_due:
                assert validation_f is not None and train_probe_reader is not None
                n_eval_tokens = probe_tokens
                probe_readers: dict[str, ShardReader] = {}
                probe_readers["train_probe"] = train_probe_reader
                probe_readers.update(validation_readers)

                all_eval_metrics: dict[str, ReconMetrics] = {}
                eval_ckpt_dir = out_dir / "eval_checkpoints"
                eval_ckpt_dir.mkdir(parents=True, exist_ok=True)

                for split_name, reader in probe_readers.items():
                    metrics, arrays = streaming_reconstruction_metrics(
                        sae,
                        _validation_batches(
                            reader,
                            n_tokens=n_eval_tokens,
                            batch_size=cfg.batch_size,
                            input_scale=normalization_scale,
                            input_center=activation_center,
                            device=device,
                        ),
                        dead_threshold_tokens=n_eval_tokens,
                        return_arrays=True,
                    )
                    assert isinstance(metrics, ReconMetrics)
                    if not all(
                        math.isfinite(float(value))
                        for value in (metrics.l0, metrics.fvu, metrics.dead_pct)
                    ):
                        raise FloatingPointError(
                            f"Non-finite evaluation metric at step {step} for {split_name}: {metrics}"
                        )
                    all_eval_metrics[split_name] = metrics
                    np.savez_compressed(
                        eval_ckpt_dir / f"eval_arrays_step_{step:07d}_{split_name}.npz",
                        firing_frequency=arrays.firing_frequency,
                        l0_per_token=arrays.l0_per_token,
                        recon_err_per_token=arrays.recon_err_per_token,
                    )

                val_metrics_only = {
                    lang: m for lang, m in all_eval_metrics.items() if lang in validation_readers
                }
                if val_metrics_only and early_stopping.enabled:
                    stable, l0_ok = early_stop_monitor.update(
                        val_metrics_only,
                        target_l0=float(target_l0),
                        cfg=early_stopping,
                        eligible=step >= early_stopping.minimum_steps,
                    )
                    stop_now = (
                        early_stopping.enabled
                        and step >= early_stopping.minimum_steps
                        and early_stop_monitor.stable_intervals >= early_stopping.patience
                    )
                else:
                    stable, l0_ok, stop_now = False, True, False

                for split_name, metrics in all_eval_metrics.items():
                    row = asdict(metrics)
                    row["inactive_pct"] = row.pop("dead_pct")
                    row.update(
                        {
                            "step": step,
                            "language": split_name,
                            "target_l0": float(target_l0),
                            "probe_sample_tokens": metrics.n_tokens,
                            "inactive_sample_tokens": metrics.n_tokens,
                            "stable_interval": stable,
                            "l0_within_tolerance": l0_ok,
                            "stable_intervals": early_stop_monitor.stable_intervals,
                            "early_stop_triggered": stop_now,
                        }
                    )
                    validation_f.write(json.dumps(row) + "\n")
                validation_f.flush()
                for split_name, metrics in all_eval_metrics.items():
                    log.info(
                        "eval step=%d split=%s fvu=%.6g l0=%.4f inactive@%d=%.2f%% D_IS=%.1f u_IS=%.2f%% D50=%.2f%% D90=%.2f%%",
                        step,
                        split_name,
                        metrics.fvu,
                        metrics.l0,
                        metrics.n_tokens,
                        metrics.dead_pct,
                        metrics.d_is,
                        metrics.u_is * 100.0,
                        metrics.d_50,
                        metrics.d_90,
                    )
                refresh_plot(
                    plot_validation_curves,
                    validation_log_path,
                    out_dir / "plots",
                    title_prefix=f"{arch} d_sae={d_sae} L0*={l0_target}",
                    description=f"validation plot at step {step}",
                )
                sae.train()
            if checkpoint_due:
                # Save after validation so resume restores the latest patience state.
                final_checkpoint = write_checkpoint(step)
            if validation_due and stop_now:
                log.info(
                    "Early stopping at step %d: validation FVU plateaued for %d "
                    "evaluations and L0 stayed within %.1f%% of target",
                    step,
                    early_stop_monitor.stable_intervals,
                    100.0 * early_stopping.l0_relative_tolerance,
                )
                break

    elapsed = time.time() - t_start
    log.info("Training done in %.1fs", elapsed)

    # Auto-generate training plots so the user has something to look at without
    # remembering to run a separate command.
    refresh_plot(
        plot_training_curves,
        jsonl_path=log_path,
        out_dir=Path(out_dir) / "plots",
        target_l0=l0_target,
        title_prefix=f"{arch}  d_sae={d_sae}  L0*={l0_target}",
        description="training plots",
    )

    if final_checkpoint is None:
        raise ValueError("Training produced no checkpoint; n_steps must be positive")
    if cfg.input_normalization == "whole_vector" and isinstance(sae, JumpReLUSAE):
        export_path = write_raw_export(
            sae,
            out_dir / f"sae_export_step_{completed_step:07d}.safetensors",
            input_scale=normalization_scale,
            activation_center=activation_center,
        )
        log.info("Wrote raw-activation SAE export %s", export_path)
    return final_checkpoint
