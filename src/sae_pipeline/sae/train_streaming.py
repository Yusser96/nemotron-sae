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
import os
import time
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
    write_raw_export,
)
from sae_pipeline.sae.jumprelu import JumpReLUSAE
from sae_pipeline.sae.normalization import load_whole_vector_normalization
from sae_pipeline.sae.objective import (
    LossCoefficients,
    coefficients_at,
    inverse_simpson_warmup_steps,
    optimizer_step,
    training_loss,
)
from sae_pipeline.sae.train import (
    ObjectiveEarlyStop,
    StepLog,
    _probe_cache_signature,
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
        if normalization_dir is None:
            raise ValueError(f"{slug}: whole_vector normalisation needs a precomputed normalization_dir")
        normalization_scale = load_whole_vector_normalization(normalization_dir, d_activation=d_in).input_scale

    sae, optim = build_sae_and_optimizer(sae_cfg, arch=arch, d_in=d_in, d_sae=d_sae, device=device)
    validation_readers = {lang: ShardReader(path) for lang, path in sorted(validation_cache_dirs.items())}
    validation_cache_signature = _probe_cache_signature(validation_readers)
    ever_fired = torch.zeros(d_sae, dtype=torch.bool, device=device)
    feature_frequency = torch.zeros(d_sae, dtype=torch.float32, device=device)
    inverse_simpson_usage = torch.zeros(d_sae, dtype=torch.float32, device=device)
    early_stop = ObjectiveEarlyStop()
    step, stopped, next_milestone_index = 0, False, 0
    feature_resets, last_reset_step = 0, None

    if resume_checkpoint is not None:
        load_sae_weights(sae, resume_checkpoint.weights_path)
        assert resume_checkpoint.trainer_state_path is not None
        state = load_trainer_state(resume_checkpoint.trainer_state_path)
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
        early_stop.load_state_dict(state["early_stop"])
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
        inverse_simpson_usage=inverse_simpson_usage,
        log_f=open(out_dir / "train_log.jsonl", mode),
        validation_f=open(out_dir / "validation_log.jsonl", mode),
        early_stop=early_stop,
        step=step,
        stopped=stopped,
        next_milestone_index=next_milestone_index,
        feature_resets=feature_resets,
        last_reset_step=last_reset_step,
    )


def streaming_step(site: StreamingSite, x_raw: torch.Tensor, step: int) -> StepLog | None:
    """One optimisation step on one captured batch; returns a log row every ``log_every`` steps."""

    cfg = site.cfg
    coefficients = coefficients_at(cfg, step)
    lr = coefficients.lr * site.early_stop.lr_scale
    x = x_raw.to(dtype=torch.float32)
    if site.normalization_scale != 1.0:
        x = x * site.normalization_scale

    target_l0 = site.l0_target + reset_l0_allowance(cfg, step, site.last_reset_step)
    terms = training_loss(site.sae, x, cfg, coefficients, target_l0=target_l0, usage=site.inverse_simpson_usage)
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
            site.sae, site.optim, x, terms.x_hat.detach(), site.feature_frequency, site.ever_fired, cfg
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
    )


def _post_warmup_coefficients(cfg: SAECfg) -> LossCoefficients:
    """Full-strength penalty weights, so held-out objectives compare across warmup."""
    return coefficients_at(cfg, max(cfg.warmup_steps, cfg.l0_warmup_steps, inverse_simpson_warmup_steps(cfg)))


@dataclass
class SiteEvaluation:
    metrics: dict[str, ReconMetrics]
    objective: dict[str, float]

    @property
    def mixed_objective(self) -> float:
        return sum(self.objective.values()) / len(self.objective)


def evaluate_site(site: StreamingSite, device: str) -> SiteEvaluation:
    """Reconstruction metrics and the held-out training objective, per language.

    One pass over each validation set: every batch is scored with the
    training loss (at post-warmup coefficients) on its way to the metrics.
    """

    coefficients = _post_warmup_coefficients(site.cfg)
    metrics: dict[str, ReconMetrics] = {}
    objective: dict[str, float] = {}
    for language, reader in site.validation_readers.items():
        weighted_total, n_rows = 0.0, 0

        def scored(batches: Iterable[torch.Tensor]) -> Iterator[torch.Tensor]:
            nonlocal weighted_total, n_rows
            for x in batches:
                with torch.no_grad():
                    terms = training_loss(site.sae, x, site.cfg, coefficients, target_l0=float(site.l0_target))
                weighted_total += float(terms.total) * x.shape[0]
                n_rows += x.shape[0]
                yield x

        batches = _validation_batches(
            reader,
            n_tokens=site.probe_tokens,
            batch_size=site.cfg.batch_size,
            input_scale=site.normalization_scale,
            input_center=None,
            device=device,
        )
        result = streaming_reconstruction_metrics(site.sae, scored(batches), dead_threshold_tokens=site.probe_tokens)
        assert isinstance(result, ReconMetrics)
        metrics[language] = result
        objective[language] = weighted_total / n_rows
    site.sae.train()
    return SiteEvaluation(metrics=metrics, objective=objective)


def record_baseline(site: StreamingSite, device: str) -> None:
    """Evaluate the untouched warm start once and keep it for delta reporting.

    A resumed site reloads the stored baseline; it cannot be recomputed after
    training has moved the weights.
    """

    path = site.out_dir / "baseline_metrics.json"
    if path.exists():
        site.baseline = json.loads(path.read_text())
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
    log.info("%s: warm-start baseline %s", site.slug, site.baseline)


def _plot_title(site: StreamingSite) -> str:
    return f"{site.arch} d_sae={site.sae.d_sae} L0*={site.l0_target} [{site.slug}]"


def write_export(site: StreamingSite, path: Path) -> Path | None:
    """Raw-activation export, for whole-vector-normalised JumpReLU sites only."""
    if site.cfg.input_normalization != "whole_vector" or not isinstance(site.sae, JumpReLUSAE):
        return None
    return write_raw_export(site.sae, path, input_scale=site.normalization_scale)


def write_checkpoint(site: StreamingSite, step: int) -> Path:
    saved = save_checkpoint_pair(
        out_dir=site.out_dir,
        step=step,
        model_state=site.sae.state_dict(),
        trainer_state={
            "version": 1,
            "global_step": step,
            "optimizer": site.optim.state_dict(),
            "random_state": capture_random_state(),
            "ever_fired": site.ever_fired.detach().cpu(),
            "feature_frequency": site.feature_frequency.detach().cpu(),
            "inverse_simpson_usage": site.inverse_simpson_usage.detach().cpu(),
            "early_stop": site.early_stop.state_dict(),
            "stopped": site.stopped,
            "next_milestone_index": site.next_milestone_index,
            "feature_resets": site.feature_resets,
            "last_reset_step": site.last_reset_step,
            "validation_cache_signature": site.validation_cache_signature,
        },
        keep_last=site.cfg.keep_last_checkpoints,
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
    write_export(site, best_dir / "sae_export.safetensors")
    (best_dir / "step.json").write_text(json.dumps({"step": step, "tokens": site.tokens_seen()}))


def evaluate_and_update(site: StreamingSite, step: int, device: str, early_stopping: EarlyStoppingCfg) -> bool:
    """Validate, update the plateau/stop state, log, and save a new best. Returns should_stop."""

    evaluation = evaluate_site(site, device)
    objective = evaluation.mixed_objective
    is_best, lr_reduced, should_stop = site.early_stop.update(
        objective, early_stopping, eligible=step >= early_stopping.minimum_steps
    )
    for language, metrics in evaluation.metrics.items():
        row = {
            **asdict(metrics),
            "step": step,
            "tokens": site.tokens_seen(),
            "language": language,
            "validation_objective": evaluation.objective[language],
            "validation_objective_mixed": objective,
            "is_best": is_best,
            "lr_reduced": lr_reduced,
            "lr_scale": site.early_stop.lr_scale,
            "checks_since_improve": site.early_stop.checks_since_improve,
            "early_stop_triggered": should_stop,
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
        site.slug, step, site.tokens_seen(), objective, site.early_stop.best,
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
    if is_best:
        _write_best(site, step)
    return should_stop


class TokenStream(Protocol):
    def next_batch(self, batch_size: int) -> torch.Tensor: ...
    def state_dict(self) -> dict[str, object]: ...
    def load_state_dict(self, state: dict[str, object]) -> None: ...


def _save_stream_state(path: Path, token_stream: TokenStream, tokens_processed: int) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(
        {
            "token_stream": token_stream.state_dict(),
            "tokens_processed": tokens_processed,
            "random_state": capture_random_state(),
        },
        temporary,
    )
    os.replace(temporary, path)


def _finish(sites: list[StreamingSite], reason: str) -> None:
    log.info("Stopping %d site(s): %s", len(sites), reason)
    for site in sites:
        site.stopped = True
        write_checkpoint(site, site.step)
        write_export(site, site.out_dir / f"sae_export_step_{site.step:07d}_final.safetensors")


def state_paths(out_root: Path, name: str | None) -> tuple[Path, Path]:
    """The stream state and ``DONE`` marker of a run, or of its trainer ``name``."""
    suffix = "" if name is None else f"_{name}"
    return out_root / f"stream_state{suffix}.pt", out_root / f"DONE{suffix}"


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
    if tokens_per_fwd % batch_size or tokens_per_fwd % seq_len:
        raise ValueError("tokens_per_fwd must be a multiple of both sae.batch_size and seq_len")

    out_root = Path(out_root)
    stream_state_path, done_path = state_paths(out_root, name)
    tokens_processed = 0
    if stream_state_path.exists():
        saved = load_trainer_state(stream_state_path)
        token_stream.load_state_dict(saved["token_stream"])
        tokens_processed = int(saved["tokens_processed"])
        restore_random_state(saved["random_state"])
        log.info("Resumed the token stream after %d model tokens", tokens_processed)

    for site in sites.values():
        if site.validation_readers:
            record_baseline(site, device)

    started = time.monotonic()
    while True:
        active = [site for site in sites.values() if not site.stopped]
        if not active:
            done = True
            break
        if stream_cfg.token_cap is not None and all(site.tokens_seen() >= stream_cfg.token_cap for site in active):
            _finish(active, f"reached the {stream_cfg.token_cap}-token cap")
            done = True
            break
        if stream_cfg.stop_after_seconds is not None and time.monotonic() - started >= stream_cfg.stop_after_seconds:
            log.info("Pausing at the %.0fs time limit", stream_cfg.stop_after_seconds)
            for site in active:
                write_checkpoint(site, site.step)
            done = False
            break
        try:
            token_batch = token_stream.next_batch(tokens_per_fwd // seq_len)
        except StopIteration:
            _finish(active, "the training corpus is exhausted")
            done = True
            break
        except TimeoutError as error:
            log.info("Pausing: %s", error)
            for site in active:
                write_checkpoint(site, site.step)
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
                site.step += 1
                step = site.step
                entry = streaming_step(site, x, step)
                if entry is not None:
                    site.log_f.write(json.dumps(asdict(entry)) + "\n")

                eval_interval = site.cfg.eval_interval_steps or early_stopping.evaluation_interval_steps
                stop_now = False
                if site.validation_readers and step % eval_interval == 0:
                    stop_now = evaluate_and_update(site, step, device, early_stopping) and early_stopping.enabled
                    checkpointed = True
                    write_checkpoint(site, step)
                elif step % site.cfg.ckpt_every == 0:
                    checkpointed = True
                    write_checkpoint(site, step)

                milestones = stream_cfg.milestone_tokens
                while site.next_milestone_index < len(milestones) and (
                    site.tokens_seen() >= milestones[site.next_milestone_index]
                ):
                    milestone = milestones[site.next_milestone_index]
                    write_export(site, site.out_dir / f"sae_export_step_{step:07d}_milestone_{milestone}.safetensors")
                    site.next_milestone_index += 1
                if stop_now:
                    _finish([site], f"{site.slug} stopped improving at step {step}")
                    checkpointed = True
                    break
        if checkpointed:
            _save_stream_state(stream_state_path, token_stream, tokens_processed)

    _save_stream_state(stream_state_path, token_stream, tokens_processed)
    for site in sites.values():
        site.log_f.flush()
        site.validation_f.flush()
    if done:
        done_path.touch()
    return done
