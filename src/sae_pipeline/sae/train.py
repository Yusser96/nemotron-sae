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
from safetensors.torch import save_file

from sae_pipeline.cache.reader import ActivationBuffer
from sae_pipeline.config import SAECfg
from sae_pipeline.eval.plots import plot_training_curves
from sae_pipeline.sae.base import SparseAutoencoder
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

    log_path = out_dir / "train_log.jsonl"
    log_f = open(log_path, "w")

    t_start = time.time()
    for step in range(1, cfg.n_steps + 1):
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
                step=step, loss=float(loss), mse=float(mse),
                l0_penalty=float(l0_penalty), hard_l0=hard_l0,
                lr=lr, lambda_l0=lam, dead_pct=dead_pct,
            )
            log_f.write(json.dumps(asdict(entry)) + "\n")
            log_f.flush()
            log.info(
                "step=%d loss=%.4f mse=%.4f l0=%.1f dead=%.1f%% lr=%.2e lam=%.2e",
                step, entry.loss, entry.mse, entry.hard_l0, entry.dead_pct, lr, lam,
            )

        if step % cfg.ckpt_every == 0 or step == cfg.n_steps:
            ckpt_path = out_dir / f"sae_step_{step:07d}.safetensors"
            save_file(
                {k: v.detach().cpu().contiguous() for k, v in sae.state_dict().items()},
                str(ckpt_path),
            )
            log.info("Wrote checkpoint %s", ckpt_path)

    log_f.close()
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

    return out_dir / f"sae_step_{cfg.n_steps:07d}.safetensors"
