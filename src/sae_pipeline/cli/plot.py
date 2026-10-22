"""Re-generate plots from existing training logs and eval artifacts.

Useful when you've changed the plotting code, want different cosmetics, or just
forgot to look at training plots while a run was in flight. Reads the same files
the training and evaluate CLIs already produce — no retraining required.

Usage:
    python -m sae_pipeline.cli.plot \
        --config configs/dev.yaml \
        --layer 25 --component resid_post \
        --arch jumprelu --width 4096 --l0 30
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from sae_pipeline.config import PipelineCfg
from sae_pipeline.cli._site import add_site_arguments, cli_site, sae_variant, variant_stem
from sae_pipeline.eval.output import evaluation_directory
from sae_pipeline.eval.plots import plot_eval_summary, plot_training_curves

log = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    add_site_arguments(p)
    p.add_argument("--arch", default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--l0", type=int, default=None)
    p.add_argument("--training-only", action="store_true",
                   help="Only regenerate training plots (skip eval histograms).")
    p.add_argument("--eval-only", action="store_true",
                   help="Only regenerate eval histograms (skip training plots).")
    p.add_argument("--partition", choices=["train", "validation"], default="train")
    p.add_argument("--language")
    p.add_argument("--output-id")
    p.add_argument("--output-tag")
    args = p.parse_args()
    if args.training_only and args.eval_only:
        p.error("--training-only and --eval-only are mutually exclusive")
    if (args.partition == "validation") != bool(args.language):
        p.error("--language is required only for --partition validation")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = PipelineCfg.from_yaml(args.config)
    spec, _site = cli_site(cfg, args, p)
    arch, width, l0_target = sae_variant(cfg, args)

    stem = variant_stem(arch, width, l0_target)
    ckpt_dir = Path(cfg.sae.ckpt_dir) / cfg.run_id / spec.slug / stem
    log_dir = evaluation_directory(cfg.log.log_dir, args.output_id or cfg.run_id,
                                   spec.slug, args.partition, args.language, args.output_tag)

    if not args.eval_only:
        train_log = ckpt_dir / "train_log.jsonl"
        if train_log.exists():
            plot_training_curves(
                jsonl_path=train_log,
                out_dir=ckpt_dir / "plots",
                target_l0=l0_target,
                title_prefix=f"{arch}  d_sae={width}  L0*={l0_target}",
            )
        else:
            log.warning("No training log at %s", train_log)

    if not args.training_only:
        summary = log_dir / f"{stem}_eval.json"
        arrays = log_dir / f"{stem}_eval_arrays.npz"
        if summary.exists():
            plot_eval_summary(
                eval_json_path=summary,
                arrays_npz_path=arrays if arrays.exists() else None,
                out_dir=log_dir / f"{stem}_plots",
                target_l0=l0_target,
                title_prefix=f"{arch}  d_sae={width}  L0*={l0_target}",
            )
        else:
            log.warning("No eval summary at %s", summary)


if __name__ == "__main__":
    main()
