"""SAE training loop. Single-GPU; one job per (layer, component, arch, width, L0).

Loss: ‖x − x̂‖² + λ · (2 / L0*) · (‖f‖₀ − L0*)²
λ is linearly warmed up from 0 to its final value over `l0_warmup_steps`.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from sae_pipeline.cache.reader import ActivationBuffer
from sae_pipeline.config import SAECfg
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

log = logging.getLogger(__name__)


def build_sae(arch: str, d_in: int, d_sae: int, bandwidth: float = 0.001) -> SparseAutoencoder:
    if arch == "jumprelu":
        return JumpReLUSAE(d_in=d_in, d_sae=d_sae, bandwidth=bandwidth)
    raise NotImplementedError(f"SAE arch {arch!r} not yet implemented (registry stub).")


def cosine_warmup(step: int, peak_lr: float, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        # Linear warmup from peak_lr * 0.1 to peak_lr (Gemma Scope 2).
        return peak_lr * (0.1 + 0.9 * step / max(1, warmup_steps))
    # Cosine decay to 0.1 * peak_lr.
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(progress, 1.0)
    return peak_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


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


def quad_l0_loss(
    l0_diff: torch.Tensor,
    l0_target: int,
) -> torch.Tensor:
    """L = (2 / L0*) * (‖f‖₀ − L0*)² (Gemma Scope 2 Eq. 5 — coefficient 2/L0 only)."""
    return (2.0 / max(l0_target, 1)) * (l0_diff - float(l0_target)).pow(2)


def train_sae(
    cfg: SAECfg,
    cache_dir: str | Path,
    d_in: int,
    arch: str,
    d_sae: int,
    l0_target: int,
    out_dir: str | Path,
    device: str | None = None,
) -> Path:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(
        "Training SAE: arch=%s d_in=%d d_sae=%d l0=%d steps=%d device=%s",
        arch, d_in, d_sae, l0_target, cfg.n_steps, device,
    )

    sae = build_sae(arch, d_in=d_in, d_sae=d_sae, bandwidth=cfg.bandwidth).to(device)
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
                "activation_buffer": buffer.state_dict(),
            },
            keep_last=cfg.keep_last_checkpoints,
        )
        log.info(
            "Wrote checkpoint pair %s and %s", saved.weights_path, saved.trainer_state_path
        )
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

            # LR + λ schedule
            lr = cosine_warmup(step, cfg.lr, cfg.warmup_steps, cfg.n_steps)
            lam = linear_warmup(step, peak=1.0, warmup_steps=cfg.l0_warmup_steps)
            for g in optim.param_groups:
                g["lr"] = lr

            x_hat, f = sae(x)
            mse = (x - x_hat).pow(2).mean()
            l0_diff = sae.l0(x)  # type: ignore[attr-defined]
            l0_penalty = quad_l0_loss(l0_diff, l0_target)
            loss = mse + lam * l0_penalty

            optim.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.decoder_unit_norm:
                sae.project_decoder_grad()
            optim.step()
            if cfg.decoder_unit_norm:
                sae.renormalize_decoder()

            with torch.no_grad():
                ever_fired |= (f > 0).any(dim=0)

            if step % cfg.log_every == 0:
                with torch.no_grad():
                    hard_l0 = sae.hard_l0(x).item()  # type: ignore[attr-defined]
                    dead_pct = 100.0 * (1.0 - ever_fired.float().mean().item())
                entry = StepLog(
                    step=step, loss=float(loss.detach()), mse=float(mse.detach()),
                    l0_penalty=float(l0_penalty.detach()), hard_l0=hard_l0,
                    lr=lr, lambda_l0=lam, dead_pct=dead_pct,
                )
                log_f.write(json.dumps(asdict(entry)) + "\n")
                log_f.flush()
                log.info(
                    "step=%d loss=%.4f mse=%.4f l0=%.1f dead=%.1f%% lr=%.2e lam=%.2e",
                    step, entry.loss, entry.mse, entry.hard_l0, entry.dead_pct, lr, lam,
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
