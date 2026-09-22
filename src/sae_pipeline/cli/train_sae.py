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

from sae_pipeline.cache.manifest import CacheManifest, manifest_path_for
from sae_pipeline.config import PipelineCfg
from sae_pipeline.hooks.components import ComponentSpec
from sae_pipeline.sae.train import train_sae

log = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--component", required=True)
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
    p.add_argument("--device", default=None)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = PipelineCfg.from_yaml(args.config)
    spec = ComponentSpec.parse(args.layer, args.component)
    arch = args.arch or (cfg.sae.arch if isinstance(cfg.sae.arch, str) else cfg.sae.arch[0])
    width = args.width or (cfg.sae.d_sae if isinstance(cfg.sae.d_sae, int) else cfg.sae.d_sae[0])
    l0_target = args.l0 or (cfg.sae.l0_target if isinstance(cfg.sae.l0_target, int) else cfg.sae.l0_target[0])

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

    out_dir = (
        Path(cfg.sae.ckpt_dir) / (args.output_id or cfg.run_id) / spec.slug / f"{arch}_w{width}_l0_{l0_target}"
    )
    sae_cfg = cfg.sae
    checkpoint_updates = {}
    if args.checkpoint_filename is not None:
        checkpoint_updates["checkpoint_filename"] = args.checkpoint_filename
    if args.checkpoint_source is not None:
        checkpoint_updates["checkpoint_source"] = args.checkpoint_source
    if args.reconstruction_loss is not None:
        checkpoint_updates["reconstruction_loss"] = args.reconstruction_loss
    if args.l0_penalty_scale is not None:
        checkpoint_updates["l0_penalty_scale"] = args.l0_penalty_scale
    if args.feature_use_strategy is not None:
        checkpoint_updates["feature_use_strategy"] = args.feature_use_strategy
    if args.residual_reset_start_step is not None:
        checkpoint_updates["residual_reset_start_step"] = args.residual_reset_start_step
    if args.residual_reset_every_steps is not None:
        checkpoint_updates["residual_reset_every_steps"] = args.residual_reset_every_steps
    if args.residual_reset_max_features is not None:
        checkpoint_updates["residual_reset_max_features"] = args.residual_reset_max_features
    if args.residual_reset_delta_l0 is not None:
        checkpoint_updates["residual_reset_delta_l0"] = args.residual_reset_delta_l0
    if args.residual_reset_calibration_tokens is not None:
        checkpoint_updates["residual_reset_calibration_tokens"] = args.residual_reset_calibration_tokens
    if args.residual_reset_anneal_steps is not None:
        checkpoint_updates["residual_reset_anneal_steps"] = args.residual_reset_anneal_steps
    if args.feature_frequency_penalty_multiplier is not None:
        checkpoint_updates["feature_frequency_penalty_multiplier"] = args.feature_frequency_penalty_multiplier
    if args.seed is not None:
        checkpoint_updates["seed"] = args.seed
    if args.steps is not None:
        checkpoint_updates["n_steps"] = args.steps
    if args.l0_start is not None:
        checkpoint_updates["l0_target_start"] = args.l0_start
    if args.l0_warmup_steps is not None:
        checkpoint_updates["l0_target_warmup_steps"] = args.l0_warmup_steps
    if args.decoder_freeze_steps is not None:
        checkpoint_updates["decoder_freeze_steps"] = args.decoder_freeze_steps
    if args.activation_centering is not None:
        checkpoint_updates["activation_centering"] = args.activation_centering
    if args.active_subspace_rank is not None:
        checkpoint_updates["active_subspace_rank"] = args.active_subspace_rank
    if checkpoint_updates:
        sae_cfg = cfg.sae.model_copy(update=checkpoint_updates)
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
    )


if __name__ == "__main__":
    main()
