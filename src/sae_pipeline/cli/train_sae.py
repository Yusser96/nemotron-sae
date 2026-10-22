"""Phase B: train an SAE on a cached (layer, component) shard set.

Usage:
    python -m sae_pipeline.cli.train_sae \
        --config configs/dev.yaml \
        --layer 25 --component resid_post \
        --arch jumprelu --width 4096 --l0 30
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from sae_pipeline.cache.manifest import (
    CacheManifest,
    manifest_path_for,
    validation_dir_for,
    validation_manifest_path_for,
)
from sae_pipeline.config import PipelineCfg
from sae_pipeline.cli._site import add_site_arguments, cli_site, sae_variant, variant_stem
from sae_pipeline.hooks.components import ComponentSpec
from sae_pipeline.sae.train import train_sae

log = logging.getLogger(__name__)


def _validation_manifest_site_mismatch(
    manifest: CacheManifest, spec: ComponentSpec
) -> str | None:
    expected = (spec.layer, spec.component)
    actual = (manifest.layer, manifest.component)
    if actual == expected:
        return None
    return f"site=({manifest.layer},{manifest.component})"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    add_site_arguments(p)
    p.add_argument("--arch", default=None)
    p.add_argument("--width", type=int, default=None, help="Override SAE.d_sae")
    p.add_argument("--l0", type=int, default=None, help="Override SAE.l0_target")
    p.add_argument("--steps", type=int, default=None, help="Override SAE.n_steps")
    p.add_argument(
        "--checkpoint-filename",
        default=None,
        help="Checkpoint file within a repository (used for per-site SAE releases)",
    )
    p.add_argument(
        "--checkpoint-source",
        default=None,
        help="Checkpoint directory, file, or Hugging Face repository.",
    )
    p.add_argument(
        "--output-id",
        default=None,
        help="Output subdirectory under sae.ckpt_dir; cache lookup still uses cfg.run_id.",
    )
    p.add_argument(
        "--reconstruction-loss",
        choices=["coordinate_mean", "vector_sum"],
        default=None,
    )
    p.add_argument("--l0-penalty-scale", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--l0-start", type=float, default=None)
    p.add_argument("--l0-warmup-steps", type=int, default=None)
    p.add_argument("--decoder-freeze-steps", type=int, default=None)
    p.add_argument("--activation-centering", choices=["none", "mean"], default=None)
    p.add_argument("--active-subspace-rank", type=int, default=None)
    p.add_argument(
        "--feature-use-strategy",
        choices=["none", "frequency", "residual_reset", "frequency_residual_reset"],
        default=None,
    )
    p.add_argument("--residual-reset-start-step", type=int, default=None)
    p.add_argument("--residual-reset-every-steps", type=int, default=None)
    p.add_argument("--residual-reset-max-features", type=int, default=None)
    p.add_argument("--residual-reset-delta-l0", type=float, default=None)
    p.add_argument("--residual-reset-calibration-tokens", type=int, default=None)
    p.add_argument("--residual-reset-anneal-steps", type=int, default=None)
    p.add_argument("--feature-frequency-penalty-multiplier", type=float, default=None)
    p.add_argument("--inverse-simpson-floor", type=float, default=None)
    p.add_argument("--inverse-simpson-max-scale", type=float, default=None)
    p.add_argument("--inverse-simpson-warmup-steps", type=int, default=None)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = PipelineCfg.from_yaml(args.config)
    spec, _site = cli_site(cfg, args, p)
    arch, width, l0_target = sae_variant(cfg, args)

    cache_dir = Path(cfg.cache.cache_dir) / cfg.run_id / spec.slug
    manifest_path = manifest_path_for(cfg.cache.cache_dir, cfg.run_id, spec.slug)
    if not manifest_path.exists():
        raise SystemExit(
            f"No cache at {manifest_path}. Run cache_activations first."
        )
    manifest = CacheManifest.read(manifest_path)
    expected_tokens = cfg.data.total_tokens
    if not manifest.complete:
        raise SystemExit(
            f"Cache at {manifest_path} is incomplete ({manifest.total_tokens} tokens); "
            "training is held until the full cache is committed."
        )
    if expected_tokens is not None and manifest.total_tokens != expected_tokens:
        raise SystemExit(
            f"Cache at {manifest_path} has {manifest.total_tokens} tokens, but the "
            f"configured training budget is {expected_tokens}."
        )
    log.info("Found cache: %s (%d shards, %d tokens, d=%d)",
             manifest_path, manifest.n_shards, manifest.total_tokens, manifest.d_activation)

    validation_cache_dirs: dict[str, Path] = {}
    languages = sorted(
        {source.language for source in (cfg.data.sources or []) if source.language}
    )
    if cfg.early_stopping.enabled and len(languages) < 2:
        raise SystemExit(
            "Bilingual early stopping requires at least two configured source languages"
        )
    validation_run_id = cfg.validation_run_id or cfg.run_id
    missing_validation_languages: list[str] = []
    for language in languages:
        validation_manifest_path = validation_manifest_path_for(
            cfg.cache.cache_dir, validation_run_id, language, spec.slug
        )
        if not validation_manifest_path.is_file():
            if cfg.early_stopping.enabled:
                raise SystemExit(f"Missing validation manifest: {validation_manifest_path}")
            missing_validation_languages.append(language)
            continue
        validation_manifest = CacheManifest.read(validation_manifest_path)
        mismatches = []
        if not validation_manifest.complete:
            mismatches.append("incomplete")
        if validation_manifest.run_id != validation_run_id:
            mismatches.append(f"run_id={validation_manifest.run_id!r}")
        if validation_manifest.model != cfg.model.name:
            mismatches.append(f"model={validation_manifest.model!r}")
        site_mismatch = _validation_manifest_site_mismatch(validation_manifest, spec)
        if site_mismatch is not None:
            mismatches.append(site_mismatch)
        if validation_manifest.language != language:
            mismatches.append(f"language={validation_manifest.language!r}")
        if validation_manifest.d_activation != manifest.d_activation:
            mismatches.append(f"d_activation={validation_manifest.d_activation}")
        if validation_manifest.total_tokens < cfg.eval.trajectory_n_tokens:
            mismatches.append(
                f"tokens={validation_manifest.total_tokens} < eval.trajectory_n_tokens="
                f"{cfg.eval.trajectory_n_tokens}"
            )
        if mismatches:
            raise SystemExit(
                f"Invalid validation cache {validation_manifest_path}: "
                + "; ".join(mismatches)
            )
        validation_cache_dirs[language] = validation_dir_for(
            cfg.cache.cache_dir, validation_run_id, language, spec.slug
        )
        log.info(
            "Validation cache %s: %d tokens (%d fixed-probe tokens)",
            validation_cache_dirs[language],
            validation_manifest.total_tokens,
            cfg.eval.trajectory_n_tokens,
        )

    if validation_cache_dirs and missing_validation_languages:
        raise SystemExit(
            "Validation caches are incomplete for configured languages: "
            + ", ".join(missing_validation_languages)
        )

    out_dir = (
        Path(cfg.sae.ckpt_dir) / (args.output_id or cfg.run_id) / spec.slug / variant_stem(arch, width, l0_target)
    )
    sae_cfg = cfg.sae
    checkpoint_updates = {}
    if cfg.sae.checkpoint_filename_template is not None:
        checkpoint_updates["checkpoint_filename"] = cfg.sae.checkpoint_filename_template.format(slug=spec.slug)
    override_fields = {
        "checkpoint_filename": "checkpoint_filename",
        "checkpoint_source": "checkpoint_source",
        "reconstruction_loss": "reconstruction_loss",
        "l0_penalty_scale": "l0_penalty_scale",
        "feature_use_strategy": "feature_use_strategy",
        "residual_reset_start_step": "residual_reset_start_step",
        "residual_reset_every_steps": "residual_reset_every_steps",
        "residual_reset_max_features": "residual_reset_max_features",
        "residual_reset_delta_l0": "residual_reset_delta_l0",
        "residual_reset_calibration_tokens": "residual_reset_calibration_tokens",
        "residual_reset_anneal_steps": "residual_reset_anneal_steps",
        "feature_frequency_penalty_multiplier": "feature_frequency_penalty_multiplier",
        "seed": "seed",
        "steps": "n_steps",
        "l0": "l0_target",
        "l0_start": "l0_target_start",
        "l0_warmup_steps": "l0_target_warmup_steps",
        "decoder_freeze_steps": "decoder_freeze_steps",
        "activation_centering": "activation_centering",
        "active_subspace_rank": "active_subspace_rank",
        "inverse_simpson_floor": "inverse_simpson_floor",
        "inverse_simpson_max_scale": "inverse_simpson_max_scale",
        "inverse_simpson_warmup_steps": "inverse_simpson_warmup_steps",
    }
    for argument, field in override_fields.items():
        value = getattr(args, argument)
        if value is not None:
            checkpoint_updates[field] = value
    if checkpoint_updates:
        sae_cfg = type(cfg.sae).model_validate({**cfg.sae.model_dump(), **checkpoint_updates})
    train_sae(
        cfg=sae_cfg,
        cache_dir=cache_dir,
        d_in=manifest.d_activation,
        arch=arch,
        d_sae=width,
        l0_target=l0_target,
        out_dir=out_dir,
        device=args.device,
        normalization_dir=(
            Path(sae_cfg.normalization_dir) / cfg.run_id / spec.slug
            if sae_cfg.normalization_dir is not None
            else None
        ),
        validation_cache_dirs=validation_cache_dirs,
        validation_tokens=cfg.eval.trajectory_n_tokens,
        validation_dead_tokens=cfg.eval.trajectory_n_tokens,
        early_stopping=cfg.early_stopping,
    )


if __name__ == "__main__":
    main()
