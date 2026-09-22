"""SAE training loop. Single-GPU; one job per (layer, component, arch, width, L0)."""

from __future__ import annotations

import json
import logging
import math
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from sae_pipeline.cache.reader import ActivationBuffer, ShardReader
from sae_pipeline.config import SAECfg
from sae_pipeline.eval.metrics import dead_pct_from_fired, firing_concentration
from sae_pipeline.eval.plots import plot_training_curves
from sae_pipeline.sae.base import SparseAutoencoder
from sae_pipeline.sae.checkpoint import (
    capture_random_state,
    load_sae_weights,
    load_trainer_state,
    resolve_checkpoint,
    restore_random_state,
    save_checkpoint_pair,
)
from sae_pipeline.sae.jumprelu import JumpReLUSAE
from sae_pipeline.sae.normalization import (
    centering_path,
    normalization_path,
    resolve_activation_center,
    resolve_whole_vector_normalization,
    write_normalization,
)
from safetensors.torch import save_file

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


def cosine_warmup(step: int, peak_lr: float, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        # Linear warmup from peak_lr * 0.1 to peak_lr (Gemma Scope 2).
        return peak_lr * (0.1 + 0.9 * step / max(1, warmup_steps))
    # Cosine decay to 0.1 * peak_lr.
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(progress, 1.0)
    return peak_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def warmup_constant(step: int, peak_lr: float, warmup_steps: int) -> float:
    """Gemma Scope 2 schedule: 0.1*peak to peak, then no decay."""
    if step >= warmup_steps:
        return peak_lr
    return peak_lr * (0.1 + 0.9 * step / max(1, warmup_steps))


def linear_warmup(step: int, peak: float, warmup_steps: int) -> float:
    if step >= warmup_steps:
        return peak
    return peak * (step / max(1, warmup_steps))


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


def quad_l0_loss(
    l0_diff: torch.Tensor,
    l0_target: float,
    scale: float = 1.0,
) -> torch.Tensor:
    """Quadratic target-L0 penalty: ``(L0-L0*)² / (2 L0*)``."""
    return scale * (1.0 / (2.0 * max(l0_target, 1))) * (l0_diff - float(l0_target)).pow(2)


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
) -> Path:
    _seed_training(cfg.seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(
        "Training SAE: arch=%s d_in=%d d_sae=%d l0=%d steps=%d device=%s",
        arch, d_in, d_sae, l0_target, cfg.n_steps, device,
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
        with torch.no_grad():
            # For x_c=x-mu, shifting b_dec by -mu preserves the initial gate.
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
                "activation_buffer": buffer.state_dict(),
            },
            keep_last=cfg.keep_last_checkpoints,
        )
        log.info(
            "Wrote checkpoint pair %s and %s", saved.weights_path, saved.trainer_state_path
        )
        if cfg.input_normalization == "whole_vector" and isinstance(sae, JumpReLUSAE):
            export_path = out_dir / f"sae_export_step_{step:07d}.safetensors"
            temporary = export_path.with_name(f".{export_path.name}.tmp")
            save_file(
                sae.raw_export_state_dict(
                    normalization_scale,
                    input_center=activation_center,
                    output_center=activation_center,
                ),
                str(temporary),
            )
            temporary.replace(export_path)
            log.info("Wrote raw-activation SAE export %s", export_path)
        # The JSONL is flushed at every checkpoint step, so this plot remains
        # useful if a later allocation ends before the final checkpoint.
        try:
            plot_training_curves(
                jsonl_path=log_path,
                out_dir=out_dir / "plots",
                target_l0=l0_target,
                title_prefix=f"{arch}  d_sae={d_sae}  L0*={l0_target}",
            )
        except Exception as exc:  # plotting must never invalidate a checkpoint
            log.warning("Failed to refresh training plot at step %d: %s", step, exc)
        return saved.weights_path

    log_path = out_dir / "train_log.jsonl"
    log_mode = "a" if cfg.checkpoint_source is not None and cfg.checkpoint_mode == "resume" else "w"

    t_start = time.time()
    final_checkpoint = source_checkpoint.weights_path if source_checkpoint is not None else None
    if source_checkpoint is not None and cfg.checkpoint_mode == "resume" and start_step == cfg.n_steps:
        log.info(
            "Resume checkpoint is already at target step %d; writing it into %s",
            cfg.n_steps, out_dir,
        )
        final_checkpoint = write_checkpoint(start_step)
    with open(log_path, log_mode) as log_f:
        for step in range(start_step + 1, cfg.n_steps + 1):
            x = buffer.next_batch().to(device, dtype=torch.float32)
            if normalization_scale != 1.0:
                x = x * normalization_scale
            if activation_center is not None:
                x = x - activation_center

            # LR + λ schedule
            lr = (
                warmup_constant(step, cfg.lr, cfg.warmup_steps)
                if cfg.lr_schedule == "warmup_constant"
                else cosine_warmup(step, cfg.lr, cfg.warmup_steps, cfg.n_steps)
            )
            lam = linear_warmup(step, peak=1.0, warmup_steps=cfg.l0_warmup_steps)
            for g in optim.param_groups:
                g["lr"] = lr

            x_hat, f = sae(x)
            squared_error = (x - x_hat).pow(2)
            reconstruction_loss = (
                squared_error.sum(dim=-1).mean()
                if cfg.reconstruction_loss == "vector_sum"
                else squared_error.mean()
            )
            mse = reconstruction_loss / x.shape[-1]
            l0_diff = sae.l0(x)  # type: ignore[attr-defined]
            target_l0 = _scheduled_l0_target(step, l0_target, cfg)
            if reset_target_step is not None:
                elapsed = step - reset_target_step
                if elapsed < cfg.residual_reset_anneal_steps:
                    target_l0 += cfg.residual_reset_delta_l0 * (
                        1.0 - elapsed / max(1, cfg.residual_reset_anneal_steps)
                    )
            l0_penalty = quad_l0_loss(l0_diff, target_l0)

            with torch.no_grad():
                fired = (f.detach() > 0).float()
                batch_frequency = fired.mean(dim=0)
                feature_frequency.mul_(cfg.feature_frequency_ema_decay).add_(
                    batch_frequency, alpha=1.0 - cfg.feature_frequency_ema_decay
                )

            feature_use_penalty = reconstruction_loss.new_zeros(())
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
                        * reconstruction_loss.detach()
                    )
                    frequency_penalty_scale = float(
                        (target_penalty / weighted_activation.detach().clamp_min(1.0e-8)).item()
                    )
                feature_use_penalty = frequency_penalty_scale * weighted_activation

            loss = (
                reconstruction_loss
                + lam * cfg.l0_penalty_scale * l0_penalty
                + feature_use_penalty
            )

            optim.zero_grad(set_to_none=True)
            loss.backward()
            with torch.no_grad():
                z = sae.gate_input(x)  # type: ignore[attr-defined]
                ste_window_pct = 100.0 * (
                    (z - sae.theta).abs() < (cfg.bandwidth / 2.0)  # type: ignore[attr-defined]
                ).float().mean().item()
                theta_grad = getattr(sae, "log_theta").grad
                encoder_grad = getattr(sae, "W_enc").grad
                theta_grad_norm = 0.0 if theta_grad is None else theta_grad.norm().item()
                encoder_grad_norm = 0.0 if encoder_grad is None else encoder_grad.norm().item()
            if cfg.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(sae.parameters(), cfg.gradient_clip_norm)
            if step <= cfg.decoder_freeze_steps:
                sae.W_dec.grad = None
            if cfg.decoder_unit_norm:
                if step > cfg.decoder_freeze_steps:
                    sae.project_decoder_grad()
            optim.step()
            if cfg.decoder_unit_norm:
                sae.renormalize_decoder()

            with torch.no_grad():
                ever_fired |= (f > 0).any(dim=0)

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
                with torch.no_grad():
                    hard_l0 = sae.hard_l0(x).item()  # type: ignore[attr-defined]
                    dead_pct = dead_pct_from_fired(ever_fired)
                    concentration = firing_concentration(feature_frequency)
                entry = StepLog(
                    step=step, loss=float(loss.detach()), mse=float(mse.detach()),
                    l0_penalty=float(l0_penalty.detach()), hard_l0=hard_l0,
                    lr=lr, lambda_l0=lam, dead_pct=dead_pct,
                    ste_window_pct=ste_window_pct,
                    theta_grad_norm=theta_grad_norm,
                    encoder_grad_norm=encoder_grad_norm,
                    reconstruction_loss=float(reconstruction_loss.detach()),
                    feature_use_penalty=float(feature_use_penalty.detach()),
                    active_features=concentration.active_features,
                    top1_activation_mass_pct=concentration.top1_activation_mass_pct,
                    max_firing_frequency=concentration.max_firing_frequency,
                    feature_resets=feature_resets,
                    target_l0=target_l0,
                )
                log_f.write(json.dumps(asdict(entry)) + "\n")
                log_f.flush()
                log.info(
                    "step=%d loss=%.4f recon=%.4f l0=%.1f dead=%.1f%% active=%d top1=%.1f%% lr=%.2e lam=%.2e",
                    step, entry.loss, entry.reconstruction_loss, entry.hard_l0,
                    entry.dead_pct, entry.active_features, entry.top1_activation_mass_pct, lr, lam,
                )

            if step % cfg.ckpt_every == 0 or step == cfg.n_steps:
                final_checkpoint = write_checkpoint(step)

    elapsed = time.time() - t_start
    log.info("Training done in %.1fs", elapsed)

    # Auto-generate training plots so the user has something to look at without
    # remembering to run a separate command.
    try:
        plot_dir = Path(out_dir) / "plots"
        plot_training_curves(
            jsonl_path=log_path,
            out_dir=plot_dir,
            target_l0=l0_target,
            title_prefix=f"{arch}  d_sae={d_sae}  L0*={l0_target}",
        )
    except Exception as e:  # plotting is non-load-bearing; never fail training over it
        log.warning("Failed to generate training plots: %s", e)

    if final_checkpoint is None:
        raise ValueError("Training produced no checkpoint; n_steps must be positive")
    return final_checkpoint
