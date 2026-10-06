"""Multi-site SAE training from live model activations (no training cache).

Instead of ``n_steps`` over a fixed activation cache (see ``sae.train``), the
frozen model runs continuously over fresh documents and every configured site
trains from the same forward pass, so no training activation is read twice.
Only a fixed validation cache lives on disk. The loss, update and log rows are
the shared helpers of ``sae.objective`` and ``sae.train``; recipe options that
need an activation cache (feature-use strategies, centring, an active
subspace) are rejected.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

import torch

from sae_pipeline.cache.reader import ShardReader
from sae_pipeline.config import SAECfg
from sae_pipeline.sae.base import SparseAutoencoder
from sae_pipeline.sae.checkpoint import (
    load_sae_weights,
    load_trainer_state,
    resolve_checkpoint,
)
from sae_pipeline.sae.jumprelu import JumpReLUSAE
from sae_pipeline.sae.normalization import load_whole_vector_normalization
from sae_pipeline.sae.objective import coefficients_at, optimizer_step, training_loss
from sae_pipeline.sae.train import (
    ObjectiveEarlyStop,
    StepLog,
    _probe_cache_signature,
    build_sae,
    record_firing,
    step_log,
)

log = logging.getLogger(__name__)


@dataclass
class StreamingSite:
    """One site's SAE, optimiser and bookkeeping inside a streaming run."""

    slug: str
    cfg: SAECfg
    sae: SparseAutoencoder
    optim: torch.optim.Optimizer
    arch: str
    l0_target: int
    normalization_scale: float
    out_dir: Path
    validation_readers: dict[str, ShardReader]
    validation_cache_signature: dict[str, dict[str, object]]
    probe_tokens: int
    ever_fired: torch.Tensor
    feature_frequency: torch.Tensor
    log_f: IO[str]
    validation_f: IO[str]
    early_stop: ObjectiveEarlyStop = field(default_factory=ObjectiveEarlyStop)
    step: int = 0
    stopped: bool = False
    next_milestone_index: int = 0

    @property
    def log_path(self) -> Path:
        return self.out_dir / "train_log.jsonl"

    @property
    def validation_log_path(self) -> Path:
        return self.out_dir / "validation_log.jsonl"

    def tokens_seen(self) -> int:
        return self.step * self.cfg.batch_size

    def close(self) -> None:
        self.log_f.close()
        self.validation_f.close()


def _reject_unsupported(cfg: SAECfg, slug: str) -> None:
    if cfg.feature_use_strategy != "none":
        raise NotImplementedError(
            f"{slug}: streaming training does not implement feature_use_strategy {cfg.feature_use_strategy!r}"
        )
    if cfg.activation_centering != "none":
        raise NotImplementedError(f"{slug}: streaming training does not implement activation_centering")
    if cfg.active_subspace_rank is not None:
        raise NotImplementedError(f"{slug}: streaming training does not implement active_subspace_rank")
    if cfg.lr_schedule == "cosine_decay":
        raise ValueError(f"{slug}: cosine_decay needs a fixed n_steps horizon, which streaming does not have")


def init_streaming_site(
    *,
    slug: str,
    sae_cfg: SAECfg,
    d_in: int,
    arch: str,
    d_sae: int,
    l0_target: int,
    out_dir: str | Path,
    device: str,
    validation_cache_dirs: dict[str, str | Path],
    probe_tokens: int,
    normalization_dir: str | Path | None = None,
    checkpoint_filename: str | None = None,
    resume_checkpoint=None,
) -> StreamingSite:
    """Build one site from its warm start, or from its own checkpoint when resuming."""

    _reject_unsupported(sae_cfg, slug)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    normalization_scale = 1.0
    if sae_cfg.input_normalization == "whole_vector":
        if normalization_dir is None:
            raise ValueError(f"{slug}: whole_vector normalisation needs a precomputed normalization_dir")
        normalization_scale = load_whole_vector_normalization(normalization_dir, d_activation=d_in).input_scale

    sae = build_sae(
        arch,
        d_in=d_in,
        d_sae=d_sae,
        bandwidth=sae_cfg.bandwidth,
        pre_encoder_bias=(sae_cfg.pre_encoder_bias or sae_cfg.input_normalization == "whole_vector"),
    ).to(device)
    optim = torch.optim.Adam(
        sae.parameters(), lr=sae_cfg.lr, betas=(sae_cfg.adam_beta1, sae_cfg.adam_beta2), eps=sae_cfg.adam_eps
    )
    validation_readers = {lang: ShardReader(path) for lang, path in sorted(validation_cache_dirs.items())}
    validation_cache_signature = _probe_cache_signature(validation_readers)
    ever_fired = torch.zeros(d_sae, dtype=torch.bool, device=device)
    feature_frequency = torch.zeros(d_sae, dtype=torch.float32, device=device)
    early_stop = ObjectiveEarlyStop()
    step, stopped, next_milestone_index = 0, False, 0

    if resume_checkpoint is not None:
        load_sae_weights(sae, resume_checkpoint.weights_path)
        assert resume_checkpoint.trainer_state_path is not None
        state = load_trainer_state(resume_checkpoint.trainer_state_path)
        saved_signature = state.get("validation_cache_signature")
        if saved_signature is not None and saved_signature != validation_cache_signature:
            raise ValueError(f"{slug}: validation caches differ from the resume checkpoint")
        step = int(state["global_step"])
        optim.load_state_dict(state["optimizer"])
        ever_fired.copy_(state["ever_fired"].to(device=device, dtype=torch.bool))
        feature_frequency.copy_(state["feature_frequency"].to(device=device, dtype=torch.float32))
        early_stop.load_state_dict(state["early_stop"])
        stopped = bool(state.get("stopped", False))
        next_milestone_index = int(state.get("next_milestone_index", 0))
        log.info("%s: resumed at step %d", slug, step)
    else:
        if sae_cfg.checkpoint_source is None:
            raise ValueError(f"{slug}: streaming training needs sae.checkpoint_source as a warm start")
        source = resolve_checkpoint(
            sae_cfg.checkpoint_source,
            filename=checkpoint_filename,
            revision=sae_cfg.checkpoint_revision,
            require_trainer_state=False,
        )
        load_sae_weights(sae, source.weights_path)
        if sae_cfg.checkpoint_format == "raw_export":
            if not isinstance(sae, JumpReLUSAE) or sae_cfg.input_normalization != "whole_vector":
                raise ValueError(f"{slug}: raw_export warm starts need a JumpReLU SAE with whole_vector input")
            sae.unfold_raw_export(normalization_scale)
        log.info("%s: fine-tuning %s with a fresh optimizer", slug, source.weights_path)

    mode = "a" if resume_checkpoint is not None else "w"
    return StreamingSite(
        slug=slug,
        cfg=sae_cfg,
        sae=sae,
        optim=optim,
        arch=arch,
        l0_target=l0_target,
        normalization_scale=normalization_scale,
        out_dir=out_dir,
        validation_readers=validation_readers,
        validation_cache_signature=validation_cache_signature,
        probe_tokens=probe_tokens,
        ever_fired=ever_fired,
        feature_frequency=feature_frequency,
        log_f=open(out_dir / "train_log.jsonl", mode),
        validation_f=open(out_dir / "validation_log.jsonl", mode),
        early_stop=early_stop,
        step=step,
        stopped=stopped,
        next_milestone_index=next_milestone_index,
    )


def streaming_step(site: StreamingSite, x_raw: torch.Tensor, step: int) -> StepLog | None:
    """One optimisation step on one captured batch; returns a log row every ``log_every`` steps."""

    cfg = site.cfg
    coefficients = coefficients_at(cfg, step)
    lr = coefficients.lr * site.early_stop.lr_scale
    x = x_raw.to(dtype=torch.float32)
    if site.normalization_scale != 1.0:
        x = x * site.normalization_scale

    target_l0 = float(site.l0_target)
    terms = training_loss(site.sae, x, cfg, coefficients, target_l0=target_l0)
    if not torch.isfinite(terms.total):
        raise FloatingPointError(f"{site.slug}: non-finite training loss at step {step}")
    record_firing(
        terms.f,
        feature_frequency=site.feature_frequency,
        ever_fired=site.ever_fired,
        ema_decay=cfg.feature_frequency_ema_decay,
    )
    gradient_stats = optimizer_step(site.sae, site.optim, terms.total, x, cfg, step=step, lr=lr)
    if step % cfg.log_every:
        return None
    return step_log(
        site.sae,
        x,
        step=step,
        loss=terms.total,
        terms=terms,
        gradient_stats=gradient_stats,
        lr=lr,
        lambda_l0=coefficients.lambda_l0,
        target_l0=target_l0,
        ever_fired=site.ever_fired,
        feature_frequency=site.feature_frequency,
    )
