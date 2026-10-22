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

import json
import logging
import math
import os
import time
from time import time_ns
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import IO, Protocol

import torch
from safetensors.torch import save_file

from sae_pipeline.cache.reader import ShardReader
from sae_pipeline.config import EarlyStoppingCfg, SAECfg, StreamCfg
from sae_pipeline.eval.metrics import ReconMetrics, streaming_reconstruction_metrics
from sae_pipeline.eval.plots import plot_training_curves, plot_validation_curves, refresh_plot
from sae_pipeline.extraction.base import ActivationExtractor
from sae_pipeline.sae.base import SparseAutoencoder
from sae_pipeline.sae.checkpoint import (
    capture_random_state,
    load_sae_weights,
    load_trainer_state,
    resolve_checkpoint,
    restore_random_state,
    save_checkpoint_pair,
    prune_checkpoints,
    write_raw_export,
)
from sae_pipeline.sae.jumprelu import JumpReLUSAE
from sae_pipeline.sae.normalization import load_whole_vector_normalization, normalization_path, write_normalization
from sae_pipeline.sae.objective import (
    LossCoefficients,
    coefficients_at,
    optimizer_step,
    training_loss,
)
from sae_pipeline.sae.train import (
    ObjectiveEarlyStop,
    StepLog,
    _probe_cache_signature,
    _seed_training,
    _scheduled_l0_target,
    _validation_batches,
    build_sae_and_optimizer,
    record_firing,
    resample_underused_features,
    reset_l0_allowance,
    residual_reset_due,
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
    # Firing rates over the Inverse-Simpson window (see objective.training_loss).
    inverse_simpson_usage: torch.Tensor
    # Tokens since each latent last fired, which defines AuxK's dead latents.
    tokens_since_fired: torch.Tensor
    log_f: IO[str]
    validation_f: IO[str]
    early_stop: ObjectiveEarlyStop = field(default_factory=ObjectiveEarlyStop)
    step: int = 0
    stopped: bool = False
    next_milestone_index: int = 0
    feature_resets: int = 0
    last_reset_step: int | None = None
    # Validation FVU/objective of the warm start, per language: reported next
    # to (never instead of) every evaluation's absolute numbers.
    baseline: dict[str, dict[str, float]] | None = None
    resume_weights_name: str | None = None
    selection_state: dict = field(default_factory=dict)
    stopping_state: dict = field(default_factory=dict)

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
    if cfg.feature_use_strategy not in {"none", "residual_reset"}:
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
    new_validation: bool = False,
) -> StreamingSite:
    """Build one site from its warm start, or from its own checkpoint when resuming.

    A checkpoint validated on other caches resumes only with ``new_validation``:
    early stopping then starts over on the new caches (their objectives do not
    compare with the old ones; the learning-rate scale reached stays), and the
    old warm-start baseline is set aside, so no deltas are reported.
    """

    _reject_unsupported(sae_cfg, slug)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    normalization_scale = 1.0
    if sae_cfg.input_normalization == "whole_vector":
        metadata_dir = normalization_dir if normalization_dir is not None else out_dir
        statistic = load_whole_vector_normalization(metadata_dir, d_activation=d_in)
        normalization_scale = statistic.input_scale
        if not math.isfinite(normalization_scale) or normalization_scale <= 0:
            raise ValueError(f"{slug}: normalization scale must be finite and positive")

    _seed_training(sae_cfg.seed)
    sae, optim = build_sae_and_optimizer(sae_cfg, arch=arch, d_in=d_in, d_sae=d_sae, device=device)
    validation_readers = {lang: ShardReader(path) for lang, path in sorted(validation_cache_dirs.items())}
    validation_cache_signature = _probe_cache_signature(validation_readers)
    ever_fired = torch.zeros(d_sae, dtype=torch.bool, device=device)
    feature_frequency = torch.zeros(d_sae, dtype=torch.float32, device=device)
    inverse_simpson_usage = torch.zeros(d_sae, dtype=torch.float32, device=device)
    tokens_since_fired = torch.zeros(d_sae, dtype=torch.int64, device=device)
    early_stop = ObjectiveEarlyStop()
    step, stopped, next_milestone_index = 0, False, 0
    feature_resets, last_reset_step = 0, None

    if resume_checkpoint is not None:
        load_sae_weights(sae, resume_checkpoint.weights_path)
        assert resume_checkpoint.trainer_state_path is not None
        state = load_trainer_state(resume_checkpoint.trainer_state_path)
        if state.get("normalization_scale", normalization_scale) != normalization_scale:
            raise ValueError(f"{slug}: normalization scale differs from the resume checkpoint")
        saved_signature = state.get("validation_cache_signature")
        validation_changed = saved_signature is not None and saved_signature != validation_cache_signature
        if validation_changed and not new_validation:
            raise ValueError(f"{slug}: validation caches differ from the resume checkpoint")
        step = int(state["global_step"])
        optim.load_state_dict(state["optimizer"])
        ever_fired.copy_(state["ever_fired"].to(device=device, dtype=torch.bool))
        feature_frequency.copy_(state["feature_frequency"].to(device=device, dtype=torch.float32))
        # Checkpoints from before the window existed: it refills over its length.
        if "inverse_simpson_usage" in state:
            inverse_simpson_usage.copy_(state["inverse_simpson_usage"].to(device=device, dtype=torch.float32))
        # Checkpoints from before AuxK: every latent counts as just fired.
        if "tokens_since_fired" in state:
            tokens_since_fired.copy_(state["tokens_since_fired"].to(device=device, dtype=torch.int64))
        early_stop.load_state_dict(state["early_stop"])
        if state.get("best_selection_policy") != "reconstruction_v1":
            early_stop.best, early_stop.checks_since_improve = None, 0
            log.warning("%s: best was not scored on reconstruction alone; reset best/plateau baseline", slug)
        if validation_changed:
            early_stop.best, early_stop.checks_since_improve = None, 0
            baseline = out_dir / "baseline_metrics.json"
            if baseline.exists():
                baseline.replace(out_dir / "baseline_metrics_previous_validation.json")
            log.warning("%s: new validation caches; early stopping starts over on them", slug)
        stopped = bool(state.get("stopped", False))
        next_milestone_index = int(state.get("next_milestone_index", 0))
        feature_resets = int(state.get("feature_resets", 0))
        last_reset_step = state.get("last_reset_step")
        log.info("%s: resumed at step %d", slug, step)
    else:
        from sae_pipeline.sae.initialization import initialize_sae
        receipt = initialize_sae(
            sae, sae_cfg, normalization_scale=normalization_scale,
            checkpoint_filename=checkpoint_filename,
        )
        (out_dir / "initialization.json").write_text(json.dumps(receipt, indent=2))
        log.info("%s: %s initialization, %d reset latents, fresh optimizer",
                 slug, receipt["mode"], receipt["reset_count"])

    for name, parameter in sae.named_parameters():
        if not torch.isfinite(parameter).all():
            raise ValueError(f"{slug}: non-finite checkpoint parameter {name}")

    if sae_cfg.input_normalization == "whole_vector":
        write_normalization(normalization_path(out_dir), statistic)
    if resume_checkpoint is not None:
        for filename in ("train_log.jsonl", "validation_log.jsonl"):
            path = out_dir / filename
            if path.exists():
                rows = []
                for line in path.read_text().splitlines():
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        break  # discard an interrupted last write
                    if int(row["step"]) <= step:
                        rows.append(line)
                temporary = path.with_suffix(".tmp")
                temporary.write_text("".join(line + "\n" for line in rows))
                os.replace(temporary, path)
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
        inverse_simpson_usage=inverse_simpson_usage,
        tokens_since_fired=tokens_since_fired,
        log_f=open(out_dir / "train_log.jsonl", mode),
        validation_f=open(out_dir / "validation_log.jsonl", mode),
        early_stop=early_stop,
        step=step,
        stopped=stopped,
        next_milestone_index=next_milestone_index,
        feature_resets=feature_resets,
        last_reset_step=last_reset_step,
        selection_state={} if resume_checkpoint is None else state.get("selection_state", {}),
        stopping_state={} if resume_checkpoint is None else state.get("stopping_state", {}),
        resume_weights_name=None if resume_checkpoint is None else resume_checkpoint.weights_path.name,
    )


def streaming_step(site: StreamingSite, x_raw: torch.Tensor, step: int) -> StepLog | None:
    """One optimisation step on one captured batch; returns a log row every ``log_every`` steps."""

    cfg = site.cfg
    coefficients = coefficients_at(cfg, step)
    lr = coefficients.lr * site.early_stop.lr_scale
    x = x_raw.to(dtype=torch.float32)
    if site.normalization_scale != 1.0:
        x = x * site.normalization_scale

    target_l0 = _scheduled_l0_target(step, site.l0_target, cfg) + reset_l0_allowance(cfg, step, site.last_reset_step)
    terms = training_loss(
        site.sae,
        x,
        cfg,
        coefficients,
        target_l0=target_l0,
        usage=site.inverse_simpson_usage,
        tokens_since_fired=site.tokens_since_fired,
    )
    if not torch.isfinite(terms.total):
        raise FloatingPointError(f"{site.slug}: non-finite training loss at step {step}")
    record_firing(
        terms.f,
        feature_frequency=site.feature_frequency,
        ever_fired=site.ever_fired,
        ema_decay=cfg.feature_frequency_ema_decay,
    )
    gradient_stats = optimizer_step(site.sae, site.optim, terms.total, x, cfg, step=step, lr=lr)
    if residual_reset_due(cfg, step):
        # No cache to calibrate on (residual_reset_calibration_tokens is for
        # train_sae): new directions and thresholds come from this batch.
        assert isinstance(site.sae, JumpReLUSAE)
        site.feature_resets += resample_underused_features(
            site.sae,
            site.optim,
            x,
            terms.x_hat.detach(),
            site.feature_frequency,
            site.ever_fired,
            cfg,
            tokens_since_fired=site.tokens_since_fired,
        )
        site.last_reset_step = step
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
        feature_resets=site.feature_resets,
        dead=site.tokens_since_fired >= cfg.auxk_dead_tokens if cfg.auxk_scale > 0 else None,
    )


# Reconstruction remains independent of the optional full-training-loss stop score.
_NO_PENALTIES = LossCoefficients(lr=0.0, lambda_l0=0.0, lambda_is=0.0)


@dataclass
class SiteEvaluation:
    metrics: dict[str, ReconMetrics]
    objective: dict[str, float]
    training_objective: dict[str, float] = field(default_factory=dict)
    loss_components: dict[str, dict[str, float]] = field(default_factory=dict)
    coefficients: dict[str, float] = field(default_factory=dict)

    def stopping_score(self, score: str) -> float:
        values = self.training_objective if score == "full_training_loss" else self.objective
        if not values:
            raise ValueError(f"No validation values for stopping score {score}")
        return sum(values.values()) / len(values)

    @property
    def mixed_objective(self) -> float:
        return sum(self.objective.values()) / len(self.objective)


def evaluate_site(site: StreamingSite, device: str) -> SiteEvaluation:
    """Score fixed probes without mutating training statistics, RNG or model mode."""
    metrics, objective, full_objective, components = {}, {}, {}, {}
    coefficients = coefficients_at(site.cfg, site.step)
    target = _scheduled_l0_target(site.step, site.l0_target, site.cfg) + reset_l0_allowance(
        site.cfg, site.step, site.last_reset_step
    )
    rng = capture_random_state()
    modes = [(module, module.training) for module in site.sae.modules()]
    try:
        for language, reader in site.validation_readers.items():
            totals = dict(reconstruction=0.0, sparsity_raw=0.0, sparsity_weighted=0.0,
                          is_weighted=0.0, auxk_weighted=0.0, total=0.0)
            n_rows = 0
            usage = site.inverse_simpson_usage.detach().clone()
            dead = site.tokens_since_fired.detach().clone()

            def scored(batches: Iterable[torch.Tensor]) -> Iterator[torch.Tensor]:
                nonlocal n_rows
                for x in batches:
                    with torch.no_grad():
                        terms = training_loss(site.sae, x, site.cfg, coefficients,
                                              target_l0=target, usage=usage, tokens_since_fired=dead)
                    values = dict(reconstruction=terms.reconstruction, sparsity_raw=terms.l0_penalty,
                                  sparsity_weighted=coefficients.lambda_l0 * site.cfg.l0_penalty_scale * terms.l0_penalty,
                                  is_weighted=terms.inverse_simpson_penalty,
                                  auxk_weighted=terms.auxk_penalty, total=terms.total)
                    for key, value in values.items():
                        number = float(value)
                        if not math.isfinite(number):
                            raise FloatingPointError(f"{site.slug}: nonfinite validation {language}/{key}")
                        totals[key] += number * x.shape[0]
                    n_rows += x.shape[0]
                    yield x

            batches = _validation_batches(reader, n_tokens=site.probe_tokens, batch_size=site.cfg.batch_size,
                                          input_scale=site.normalization_scale, input_center=None, device=device)
            result = streaming_reconstruction_metrics(site.sae, scored(batches), dead_threshold_tokens=site.probe_tokens)
            if n_rows != site.probe_tokens:
                raise ValueError("Validation did not consume the fixed probe exactly")
            metrics[language] = result
            components[language] = {key: value / n_rows for key, value in totals.items()}
            objective[language] = components[language]['reconstruction']
            full_objective[language] = components[language]['total']
    finally:
        for module, mode in modes:
            module.training = mode
        restore_random_state(rng)
    return SiteEvaluation(metrics=metrics, objective=objective, training_objective=full_objective,
                          loss_components=components, coefficients={**asdict(coefficients), 'target_l0': target})


def record_baseline(site: StreamingSite, device: str) -> None:
    """Evaluate the untouched warm start once and keep it for delta reporting.

    A resumed site reloads the stored baseline; it cannot be recomputed after
    training has moved the weights.
    """

    path = site.out_dir / "baseline_metrics.json"
    if path.exists():
        site.baseline = json.loads(path.read_text())
        reconstruction_missing=site.cfg.track_reconstruction_selection and not site.selection_state
        full_missing=(site.stopping_state.get('policy',{}).get('score')=='full_training_loss'
                      and not site.stopping_state.get('best_full_objective'))
        if site.step == 0 and (reconstruction_missing or full_missing):
            evaluation=evaluate_site(site, device)
            if reconstruction_missing:
                from sae_pipeline.sae.selection import observe_selection
                observe_selection(site, evaluation, EarlyStoppingCfg(mode="objective", patience=10))
            if full_missing:
                from sae_pipeline.sae.validation_stopping import observe_full_best
                observe_full_best(site,evaluation)
        return
    if site.step > 0:
        log.warning("%s: no baseline from the start of this run; deltas are not reported", site.slug)
        return
    evaluation = evaluate_site(site, device)
    site.baseline = {
        language: {"fvu": evaluation.metrics[language].fvu, "objective": evaluation.objective[language]}
        for language in evaluation.metrics
    }
    path.write_text(json.dumps(site.baseline))
    _write_diagnostic(site, evaluation, "initial")
    if site.stopping_state.get('policy',{}).get('score')=='full_training_loss':
        from sae_pipeline.sae.validation_stopping import observe_full_best
        observe_full_best(site,evaluation)
    if site.cfg.track_reconstruction_selection:
        from sae_pipeline.sae.selection import observe_selection
        observe_selection(site, evaluation, EarlyStoppingCfg(mode="objective", patience=10))
    log.info("%s: warm-start baseline %s", site.slug, site.baseline)


def _write_diagnostic(site: StreamingSite, evaluation: SiteEvaluation, phase: str, **extra) -> None:
    """Full boundary metrics without changing plateau or learning-rate state."""
    payload = {
        "phase": phase, "step": site.step, "tokens": site.tokens_seen(),
        "validation_objective_mixed": evaluation.mixed_objective,
        "validation_training_objective": evaluation.training_objective,
        "loss_components": evaluation.loss_components,
        "loss_coefficients": evaluation.coefficients,
        "metrics": {language: {**asdict(metrics), "validation_objective": evaluation.objective[language]}
                    for language, metrics in evaluation.metrics.items()},
        **extra,
    }
    path = site.out_dir / f"{phase}_metrics.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload))
    os.replace(temporary, path)


def _plot_title(site: StreamingSite) -> str:
    return f"{site.arch} d_sae={site.sae.d_sae} L0*={site.l0_target} [{site.slug}]"


def write_export(site: StreamingSite, path: Path) -> Path | None:
    """Raw-activation export, for whole-vector-normalised JumpReLU sites only."""
    if site.cfg.input_normalization != "whole_vector" or not isinstance(site.sae, JumpReLUSAE):
        return None
    return write_raw_export(site.sae, path, input_scale=site.normalization_scale)


def write_checkpoint(site: StreamingSite, step: int, *, snapshot_id: str | None = None) -> Path:
    saved = save_checkpoint_pair(
        out_dir=site.out_dir,
        step=step,
        model_state=site.sae.state_dict(),
        trainer_state={
            "version": 1,
            "selection_state": site.selection_state,
            "stopping_state": site.stopping_state,
            "global_step": step,
            "optimizer": site.optim.state_dict(),
            "random_state": capture_random_state(),
            "ever_fired": site.ever_fired.detach().cpu(),
            "feature_frequency": site.feature_frequency.detach().cpu(),
            "inverse_simpson_usage": site.inverse_simpson_usage.detach().cpu(),
            "tokens_since_fired": site.tokens_since_fired.detach().cpu(),
            "early_stop": site.early_stop.state_dict(),
            "best_selection_policy": "reconstruction_v1",
            "stopped": site.stopped,
            "next_milestone_index": site.next_milestone_index,
            "feature_resets": site.feature_resets,
            "last_reset_step": site.last_reset_step,
            "validation_cache_signature": site.validation_cache_signature,
            "normalization_scale": site.normalization_scale,
        },
        keep_last=site.cfg.keep_last_checkpoints,
        prune=snapshot_id is None,
        snapshot_id=snapshot_id,
    )
    site.log_f.flush()
    refresh_plot(
        plot_training_curves,
        jsonl_path=site.log_path,
        out_dir=site.out_dir / "plots",
        target_l0=site.l0_target,
        title_prefix=_plot_title(site),
        description=f"{site.slug} training plot at step {step}",
    )
    return saved.weights_path


def _write_best(site: StreamingSite, step: int) -> None:
    best_dir = site.out_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        {key: value.detach().cpu().contiguous() for key, value in site.sae.state_dict().items()},
        str(best_dir / "sae_weights.safetensors"),
    )
    if site.cfg.input_normalization == "whole_vector":
        statistic = load_whole_vector_normalization(site.out_dir, d_activation=site.sae.d_in)
        write_normalization(normalization_path(best_dir), statistic)
    write_export(site, best_dir / "sae_export.safetensors")
    (best_dir / "step.json").write_text(json.dumps({"step": step, "tokens": site.tokens_seen()}))


@dataclass
class Stop:
    """Why a site stops after an evaluation; a diverged one gets no final export."""

    reason: str
    diverged: bool = False


def finite_evaluation_error(evaluation):
    values = list(evaluation.objective.values()) + list(evaluation.training_objective.values())
    values.append(evaluation.mixed_objective)
    for metrics in evaluation.metrics.values():
        values.extend(value for value in vars(metrics).values()
                      if isinstance(value, (int, float)) and not isinstance(value, bool))
    if any(not math.isfinite(value) for value in values):
        return 'nonfinite validation objective or metric'
    return None


def threshold_warnings(site, evaluation, step):
    """Return finite warning labels; caller logs them in the normal eval cadence."""
    if finite_evaluation_error(evaluation) is not None:
        return ()
    warnings = []
    l0 = max(metrics.l0 for metrics in evaluation.metrics.values())
    if step >= site.cfg.l0_warmup_steps and l0 > 3.0 * site.l0_target:
        warnings.append('l0')
    best = site.early_stop.best
    if best is not None and evaluation.mixed_objective > 10.0 * best:
        warnings.append('objective')
    return tuple(warnings)


def _divergence(site: StreamingSite, evaluation: SiteEvaluation, step: int, cfg: EarlyStoppingCfg) -> str | None:
    """How this evaluation shows ``site`` diverging, if it does (see ``EarlyStoppingCfg``)."""
    objective = evaluation.stopping_score(cfg.score)
    error = finite_evaluation_error(evaluation)
    if error is not None:
        return error
    l0 = max(metrics.l0 for metrics in evaluation.metrics.values())
    if (
        cfg.divergence_l0_factor is not None
        and step >= site.cfg.l0_warmup_steps
        and l0 > cfg.divergence_l0_factor * site.l0_target
    ):
        return f"validation L0 {l0:.1f} above {cfg.divergence_l0_factor:g}x the target {site.l0_target}"
    best = site.early_stop.best
    if cfg.divergence_objective_factor is not None and best is not None and objective > cfg.divergence_objective_factor * best:
        return f"validation objective {objective:.4g} above {cfg.divergence_objective_factor:g}x its best {best:.4g}"
    return None


def evaluate_and_update(
    site: StreamingSite, step: int, device: str, early_stopping: EarlyStoppingCfg
) -> Stop | None:
    """Validate, update the plateau/stop state, log, and save a new best.

    Returns why the site should stop: it diverged (always checked), or it
    stopped improving (with early stopping enabled); None to go on.
    """

    evaluation = evaluate_site(site, device)
    from sae_pipeline.sae.validation_stopping import configure_stopping, eligibility_step, observe_full_best
    configure_stopping(site, early_stopping)
    objective = evaluation.stopping_score(early_stopping.score)
    divergence = _divergence(site, evaluation, step, early_stopping)
    warnings = threshold_warnings(site, evaluation, step)
    if warnings:
        log.warning("%s finite diagnostic threshold warnings at step %d: %s", site.slug, step, ",".join(warnings))
    if site.cfg.track_reconstruction_selection and divergence is None:
        from sae_pipeline.sae.selection import observe_selection
        observe_selection(site, evaluation, early_stopping)
    selection_minimum = eligibility_step(site.cfg, early_stopping)
    selection_eligible = step >= selection_minimum
    # Best exports are independent of the protected stopping baseline/patience.
    if divergence is None and early_stopping.score == "full_training_loss":
        observe_full_best(site, evaluation)
    if divergence is None and selection_eligible:
        if early_stopping.score == "full_training_loss" and not site.stopping_state.get('baseline_initialized'):
            site.early_stop.best, site.early_stop.checks_since_improve = None, 0
            site.stopping_state['baseline_initialized'] = True
        is_best, lr_reduced, should_stop = site.early_stop.update(
            objective, early_stopping, eligible=early_stopping.enabled and step >= early_stopping.minimum_steps
        )
    else:
        is_best, lr_reduced, should_stop = False, False, False
    for language, metrics in evaluation.metrics.items():
        row = {
            **asdict(metrics),
            "step": step,
            "tokens": site.tokens_seen(),
            "language": language,
            "validation_objective": evaluation.objective[language],
            "validation_objective_mixed": evaluation.mixed_objective,
            "validation_training_objective": evaluation.training_objective.get(language),
            "validation_stopping_score": objective,
            "stopping_score_kind": early_stopping.score,
            "loss_components": evaluation.loss_components.get(language, {}),
            "loss_coefficients": evaluation.coefficients,
            "is_best": is_best,
            "selection_eligible": selection_eligible,
            "selection_minimum_steps": selection_minimum,
            "lr_reduced": lr_reduced,
            "lr_scale": site.early_stop.lr_scale,
            "checks_since_improve": site.early_stop.checks_since_improve,
            "early_stop_triggered": should_stop,
            "diverged": divergence,
            "finite_l0_overshoot": float("l0" in warnings),
            "finite_objective_overshoot": float("objective" in warnings),
        }
        if site.baseline is not None and language in site.baseline:
            baseline = site.baseline[language]
            row["baseline_fvu"] = baseline["fvu"]
            row["delta_fvu_vs_initial"] = metrics.fvu - baseline["fvu"]
            row["baseline_objective"] = baseline["objective"]
            row["delta_objective_vs_initial"] = evaluation.objective[language] - baseline["objective"]
        site.validation_f.write(json.dumps(row) + "\n")
    site.validation_f.flush()
    log.info(
        "%s eval step=%d tokens=%d objective=%.6g best=%.6g stale=%d lr_scale=%.3g%s%s",
        site.slug, step, site.tokens_seen(), objective, math.nan if site.early_stop.best is None else site.early_stop.best,
        site.early_stop.checks_since_improve, site.early_stop.lr_scale,
        " [new best]" if is_best else "", " [lr reduced]" if lr_reduced else "",
    )
    refresh_plot(
        plot_validation_curves,
        site.validation_log_path,
        site.out_dir / "plots",
        title_prefix=_plot_title(site),
        description=f"{site.slug} validation plot at step {step}",
    )
    if is_best and not site.cfg.track_reconstruction_selection:
        _write_best(site, step)
    if divergence is not None:
        return Stop(f"diverged: {divergence}", diverged=True)
    if should_stop and early_stopping.enabled:
        return Stop("stopped improving")
    return None


class TokenStream(Protocol):
    def next_batch(self, batch_size: int) -> torch.Tensor: ...
    def state_dict(self) -> dict[str, object]: ...
    def load_state_dict(self, state: dict[str, object]) -> None: ...


def _save_stream_state(path: Path, token_stream: TokenStream, tokens_processed: int, checkpoints: dict[str, str]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(
        {
            "checkpoints": checkpoints,
            "token_stream": token_stream.state_dict(),
            "tokens_processed": tokens_processed,
            "random_state": capture_random_state(),
        },
        temporary,
    )
    os.replace(temporary, path)


def _finish(sites: list[StreamingSite], reason: str, *, final_export: bool = True,
            early_stopping: EarlyStoppingCfg | None = None) -> None:
    log.info("Stopping %d site(s): %s", len(sites), reason)
    for site in sites:
        export_healthy = final_export
        divergence = None
        if site.validation_readers:
            device = str(next(site.sae.parameters()).device)
            evaluation = evaluate_site(site, device)
            if early_stopping is not None:
                divergence = _divergence(site, evaluation, site.step, early_stopping)
                export_healthy = export_healthy and divergence is None
            _write_diagnostic(site, evaluation, "terminal", reason=reason,
                              diverged=divergence, healthy=export_healthy)
            if site.cfg.track_reconstruction_selection and export_healthy:
                from sae_pipeline.sae.selection import observe_selection
                observe_selection(site, evaluation, early_stopping or EarlyStoppingCfg(mode="objective", patience=10))
        site.stopped = True
        if export_healthy:
            write_export(site, site.out_dir / f"sae_export_step_{site.step:07d}_final.safetensors")
        else:
            log.warning("%s: no final export from diverged weights; its result is %s", site.slug, site.out_dir / "best")


def state_paths(out_root: Path, name: str | None) -> tuple[Path, Path]:
    """The stream state and ``DONE`` marker of a run, or of its trainer ``name``."""
    suffix = "" if name is None else f"_{name}"
    return out_root / f"stream_state{suffix}.pt", out_root / f"DONE{suffix}"


def committed_stream_checkpoints(out_root: Path, site_dirs: dict[str, Path], name: str | None = None):
    """Select exactly the checkpoint pairs published with the token cursor.

    Orphan pairs from an interrupted transaction are deliberately ignored.
    """
    path = state_paths(out_root, name)[0]
    if not path.exists():
        return {slug: None for slug in site_dirs}
    state = load_trainer_state(path)
    checkpoints = state.get("checkpoints", {})
    if set(checkpoints) != set(site_dirs):
        raise ValueError("Streaming resume requires the same sites and a committed snapshot; legacy states cannot resume exactly")
    return {slug: resolve_checkpoint(directory, filename=checkpoints[slug], require_trainer_state=True)
            for slug, directory in site_dirs.items()}


def run_streaming_training(
    *,
    sites: dict[str, StreamingSite],
    token_stream: TokenStream,
    extractor: ActivationExtractor,
    stream_cfg: StreamCfg,
    early_stopping: EarlyStoppingCfg,
    tokens_per_fwd: int,
    seq_len: int,
    device: str,
    out_root: str | Path,
    name: str | None = None,
) -> bool:
    """Train every site from one continuous stream of model activations.

    ``extractor`` maps each ``(sequences, seq_len)`` token batch to
    ``{slug: (tokens, d_in)}`` activations for every site. Returns True once the run is
    complete (every site stopped, hit the token cap, or the corpus ran out);
    False when it paused at ``stream_cfg.stop_after_seconds``, or because the
    stream timed out waiting for data, and should be resumed. State is
    checkpointed so a resumed run repeats no tokens. Trainers of different
    sites sharing ``out_root`` each need a ``name``: the stream state and the
    ``DONE`` marker are per name (see ``state_paths``).
    """

    if early_stopping.enabled and early_stopping.mode != "objective":
        raise ValueError("Streaming training supports only early_stopping.mode='objective'")
    batch_sizes = {site.cfg.batch_size for site in sites.values()}
    if len(batch_sizes) != 1:
        raise ValueError(f"All streamed sites must share one batch_size; got {batch_sizes}")
    batch_size = batch_sizes.pop()
    if stream_cfg.token_cap is not None and stream_cfg.token_cap % batch_size:
        raise ValueError("stream.token_cap must be a multiple of sae.batch_size")
    if tokens_per_fwd <= 0 or seq_len <= 0:
        raise ValueError("tokens_per_fwd and seq_len must be positive")
    if tokens_per_fwd % batch_size or tokens_per_fwd % seq_len:
        raise ValueError("tokens_per_fwd must be a multiple of both sae.batch_size and seq_len")

    out_root = Path(out_root)
    stream_state_path, done_path = state_paths(out_root, name)
    tokens_processed = 0
    if stream_state_path.exists():
        saved = load_trainer_state(stream_state_path)
        if set(saved.get("checkpoints", {})) != set(sites):
            raise ValueError("Streaming resume requires the same sites and a committed snapshot; legacy states cannot resume exactly")
        for slug, site in sites.items():
            committed = resolve_checkpoint(site.out_dir, filename=saved["checkpoints"][slug], require_trainer_state=True)
            if committed.step != site.step or committed.weights_path.name != site.resume_weights_name:
                raise ValueError(f"{slug}: loaded step does not match the committed stream snapshot")
        token_stream.load_state_dict(saved["token_stream"])
        tokens_processed = int(saved["tokens_processed"])
        restore_random_state(saved["random_state"])
        log.info("Resumed the token stream after %d model tokens", tokens_processed)

    def commit_snapshot():
        # Unique names keep the previous transaction immutable until the cursor
        # and every site's pair have been atomically published together.
        snapshot_id = str(time_ns())
        checkpoints = {slug: write_checkpoint(site, site.step, snapshot_id=snapshot_id).name
                       for slug, site in sites.items()}
        _save_stream_state(stream_state_path, token_stream, tokens_processed, checkpoints)
        for slug, site in sites.items():
            prune_checkpoints(site.out_dir, keep_last=site.cfg.keep_last_checkpoints,
                              protected_weights=checkpoints[slug])
            from sae_pipeline.sae.validation_stopping import prune_full_best
            prune_full_best(site)
            if site.cfg.track_reconstruction_selection:
                from sae_pipeline.sae.selection import prune_selection
                prune_selection(site)

    from sae_pipeline.sae.validation_stopping import configure_stopping
    for site in sites.values():
        configure_stopping(site, early_stopping)
        if not early_stopping.enabled:
            site.early_stop.lr_scale = 1.0
        if site.validation_readers:
            record_baseline(site, device)

    started = time.monotonic()
    while True:
        active = [site for site in sites.values() if not site.stopped]
        if not active:
            done = True
            break
        if stream_cfg.token_cap is not None and all(site.tokens_seen() >= stream_cfg.token_cap for site in active):
            _finish(active, f"reached the {stream_cfg.token_cap}-token cap", early_stopping=early_stopping)
            done = True
            break
        if stream_cfg.stop_after_seconds is not None and time.monotonic() - started >= stream_cfg.stop_after_seconds:
            log.info("Pausing at the %.0fs time limit", stream_cfg.stop_after_seconds)
            done = False
            break
        try:
            token_batch = token_stream.next_batch(tokens_per_fwd // seq_len)
        except StopIteration:
            _finish(active, "the training corpus is exhausted", early_stopping=early_stopping)
            done = True
            break
        except TimeoutError as error:
            log.info("Pausing: %s", error)
            done = False
            break

        captured = extractor.extract(token_batch).activations
        tokens_processed += tokens_per_fwd
        checkpointed = False
        for site in active:
            x_all = captured[site.slug]
            if x_all.shape[0] != tokens_per_fwd:
                raise RuntimeError(f"{site.slug}: captured {x_all.shape[0]} activations, expected {tokens_per_fwd}")
            # Rows of one forward pass come in sequence order; shuffle them
            # locally so no SAE batch is a contiguous run of positions.
            x_all = x_all[torch.randperm(x_all.shape[0], device=x_all.device)]
            for x in x_all.split(batch_size):
                if stream_cfg.token_cap is not None and site.tokens_seen() >= stream_cfg.token_cap:
                    _finish([site], f"reached the {stream_cfg.token_cap}-token cap", early_stopping=early_stopping)
                    checkpointed = True
                    break
                site.step += 1
                step = site.step
                entry = streaming_step(site, x, step)
                if entry is not None:
                    site.log_f.write(json.dumps(asdict(entry)) + "\n")

                eval_interval = site.cfg.eval_interval_steps or early_stopping.evaluation_interval_steps
                stop = None
                if site.validation_readers and step % eval_interval == 0:
                    stop = evaluate_and_update(site, step, device, early_stopping)
                    checkpointed = True
                elif step % site.cfg.ckpt_every == 0:
                    checkpointed = True

                milestones = stream_cfg.milestone_tokens
                while site.next_milestone_index < len(milestones) and (
                    site.tokens_seen() >= milestones[site.next_milestone_index]
                ):
                    milestone = milestones[site.next_milestone_index]
                    write_export(site, site.out_dir / f"sae_export_step_{step:07d}_milestone_{milestone}.safetensors")
                    site.next_milestone_index += 1
                if stop is None and stream_cfg.token_cap is not None and site.tokens_seen() >= stream_cfg.token_cap:
                    stop = Stop(f"reached the {stream_cfg.token_cap}-token cap")
                if stop is not None:
                    _finish([site], f"{site.slug} {stop.reason} at step {step}", final_export=not stop.diverged,
                            early_stopping=early_stopping)
                    checkpointed = True
                    break
        if checkpointed:
            commit_snapshot()

    commit_snapshot()
    for site in sites.values():
        site.log_f.flush()
        site.validation_f.flush()
    if done:
        done_path.touch()
    return done
