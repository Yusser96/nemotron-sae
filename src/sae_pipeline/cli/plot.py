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
from sae_pipeline.eval.plots import plot_eval_summary, plot_training_curves
from sae_pipeline.hooks.components import ComponentSpec

log = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--component", required=True)
    p.add_argument("--arch", default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--l0", type=int, default=None)
    p.add_argument("--training-only", action="store_true",
                   help="Only regenerate training plots (skip eval histograms).")
    p.add_argument("--eval-only", action="store_true",
                   help="Only regenerate eval histograms (skip training plots).")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = PipelineCfg.from_yaml(args.config)
    spec = ComponentSpec.parse(args.layer, args.component)
    arch = args.arch or (cfg.sae.arch if isinstance(cfg.sae.arch, str) else cfg.sae.arch[0])
    width = args.width or (cfg.sae.d_sae if isinstance(cfg.sae.d_sae, int) else cfg.sae.d_sae[0])
    l0_target = args.l0 or (cfg.sae.l0_target if isinstance(cfg.sae.l0_target, int) else cfg.sae.l0_target[0])

    stem = f"{arch}_w{width}_l0_{l0_target}"
    ckpt_dir = Path(cfg.sae.ckpt_dir) / cfg.run_id / spec.slug / stem
    log_dir = Path(cfg.log.log_dir) / cfg.run_id / spec.slug

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
