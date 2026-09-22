"""Evaluate a trained SAE on its activation cache.

Usage:
    python -m sae_pipeline.cli.evaluate \
        --config configs/dev.yaml \
        --layer 25 --component resid_post \
        --arch jumprelu --width 4096 --l0 30
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from sae_pipeline.cache.manifest import CacheManifest, validation_dir_for
from sae_pipeline.cache.reader import ShardReader
from sae_pipeline.config import PipelineCfg
from sae_pipeline.eval.metrics import streaming_reconstruction_metrics
from sae_pipeline.eval.plots import plot_eval_summary
from sae_pipeline.hooks.components import ComponentSpec
from sae_pipeline.sae.checkpoint import load_sae_weights, resolve_checkpoint
from sae_pipeline.sae.normalization import normalization_path
from sae_pipeline.sae.train import build_sae

log = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--component", required=True)
    p.add_argument("--arch", default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--l0", type=int, default=None)
    p.add_argument("--checkpoint", default=None,
                   help="Specific checkpoint .safetensors. If omitted, use the latest.")
    p.add_argument("--checkpoint-source", default=None,
                   help="Checkpoint directory, file, or Hugging Face repository to evaluate.")
    p.add_argument("--checkpoint-filename", default=None,
                   help="Checkpoint file within --checkpoint-source (for per-site releases).")
    p.add_argument("--output-tag", default=None,
                   help="Optional subdirectory for keeping separate evaluation variants.")
    p.add_argument("--output-id", default=None,
                   help="Output subdirectory under log.log_dir; cache lookup still uses cfg.run_id.")
    p.add_argument("--checkpoint-format", choices=["canonical", "raw_export"], default=None,
                   help="Canonical checkpoints use cached normalisation; raw exports use raw activations.")
    p.add_argument("--partition", choices=["train", "validation"], default="train")
    p.add_argument("--language", help="Required with --partition validation.")
    p.add_argument("--device", default=None)
    args = p.parse_args()
    if args.partition == "validation" and not args.language:
        p.error("--language is required with --partition validation")
    if args.partition == "train" and args.language:
        p.error("--language is only valid with --partition validation")
    if args.checkpoint and (args.checkpoint_source or args.checkpoint_filename):
        p.error("Use either --checkpoint or --checkpoint-source/--checkpoint-filename")
    if args.checkpoint_filename and not args.checkpoint_source:
        p.error("--checkpoint-filename requires --checkpoint-source")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = PipelineCfg.from_yaml(args.config)
    spec = ComponentSpec.parse(args.layer, args.component)
    arch = args.arch or (cfg.sae.arch if isinstance(cfg.sae.arch, str) else cfg.sae.arch[0])
    width = args.width or (cfg.sae.d_sae if isinstance(cfg.sae.d_sae, int) else cfg.sae.d_sae[0])
    l0_target = args.l0 or (cfg.sae.l0_target if isinstance(cfg.sae.l0_target, int) else cfg.sae.l0_target[0])

    if args.partition == "train":
        cache_dir = Path(cfg.cache.cache_dir) / cfg.run_id / spec.slug
    else:
        validation_run_id = cfg.validation_run_id or cfg.run_id
        cache_dir = validation_dir_for(
            cfg.cache.cache_dir, validation_run_id, args.language, spec.slug
        )
    manifest_path = cache_dir / "manifest.json"
    manifest = CacheManifest.read(manifest_path)
    reader = ShardReader(cache_dir)

    ckpt_dir = (
        Path(cfg.sae.ckpt_dir) / cfg.run_id / spec.slug / f"{arch}_w{width}_l0_{l0_target}"
    )
    checkpoint_source = args.checkpoint or args.checkpoint_source or ckpt_dir
    checkpoint_filename = (
        None
        if args.checkpoint
        else args.checkpoint_filename
        if args.checkpoint_source
        else cfg.sae.checkpoint_filename
    )
    resolved = resolve_checkpoint(
        checkpoint_source,
        filename=checkpoint_filename,
        revision=cfg.sae.checkpoint_revision,
    )
    ckpt = resolved.weights_path

    checkpoint_format = args.checkpoint_format or (
        "raw_export" if cfg.sae.checkpoint_format == "raw_export" and args.checkpoint_source else "canonical"
    )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    sae = build_sae(
        arch,
        d_in=manifest.d_activation,
        d_sae=width,
        bandwidth=cfg.sae.bandwidth,
        pre_encoder_bias=checkpoint_format == "canonical" and (
            cfg.sae.pre_encoder_bias or cfg.sae.input_normalization == "whole_vector"
        ),
    ).to(device)
    load_sae_weights(sae, ckpt)
    sae.eval()
    log.info("Loaded SAE from %s", ckpt)

    input_scale = 1.0
    input_center = None
    if checkpoint_format == "canonical" and cfg.sae.input_normalization == "whole_vector":
        norm_path = normalization_path(ckpt.parent)
        if not norm_path.exists():
            raise FileNotFoundError(
                f"Canonical checkpoint requires normalisation metadata at {norm_path}"
            )
        with norm_path.open() as handle:
            input_scale = float(json.load(handle)["input_scale"])
        centre_path = ckpt.parent / "centering.json"
        if centre_path.exists():
            with centre_path.open() as handle:
                input_center = torch.tensor(
                    json.load(handle)["mean"], device=device, dtype=torch.float32
                )
            if input_center.numel() != manifest.d_activation:
                raise ValueError(f"Centring metadata at {centre_path} has the wrong dimension")

    target = cfg.eval.fvu_n_tokens
    batch_size = min(cfg.sae.batch_size, 4096)

    def batches():
        seen = 0
        for shard in reader.iter_shards():
            for start in range(0, shard.shape[0], batch_size):
                if seen >= target:
                    return
                batch = shard[start : start + min(batch_size, target - seen)]
                seen += batch.shape[0]
                batch = batch.to(device, dtype=torch.float32)
                if input_scale != 1.0:
                    batch = batch * input_scale
                if input_center is not None:
                    batch = batch - input_center
                yield batch

    metrics, arrays = streaming_reconstruction_metrics(
        sae,
        batches(),
        dead_threshold_tokens=cfg.eval.dead_n_tokens,
        return_arrays=True,
    )
    log.info("Eval: %s", metrics)

    out_dir = Path(cfg.log.log_dir) / (args.output_id or cfg.run_id) / spec.slug / args.partition
    if args.language:
        out_dir = out_dir / args.language
    if args.output_tag:
        out_dir = out_dir / args.output_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{arch}_w{width}_l0_{l0_target}"

    summary_path = out_dir / f"{stem}_eval.json"
    payload = {
        "checkpoint": str(ckpt),
        "manifest": str(manifest_path),
        "arch": arch,
        "d_sae": width,
        "l0_target": l0_target,
        "partition": args.partition,
        "language": args.language,
        "output_tag": args.output_tag,
        "output_id": args.output_id,
        "checkpoint_format": checkpoint_format,
        "input_scale": input_scale,
        "input_centered": input_center is not None,
        **asdict(metrics),
    }
    with open(summary_path, "w") as f:
        json.dump(payload, f, indent=2)
    log.info("Wrote eval summary to %s", summary_path)

    arrays_path = out_dir / f"{stem}_eval_arrays.npz"
    np.savez_compressed(
        arrays_path,
        firing_frequency=arrays.firing_frequency,
        l0_per_token=arrays.l0_per_token,
        recon_err_per_token=arrays.recon_err_per_token,
    )
    log.info("Wrote eval arrays to %s", arrays_path)

    plot_dir = out_dir / f"{stem}_plots"
    try:
        plot_eval_summary(
            eval_json_path=summary_path,
            arrays_npz_path=arrays_path,
            out_dir=plot_dir,
            target_l0=l0_target,
            title_prefix=f"{arch}  d_sae={width}  L0*={l0_target}",
        )
    except Exception as e:
        log.warning("Failed to generate eval plots: %s", e)


if __name__ == "__main__":
    main()
