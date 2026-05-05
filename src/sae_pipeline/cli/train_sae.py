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
    log.info("Found cache: %s (%d shards, %d tokens, d=%d)",
             manifest_path, manifest.n_shards, manifest.total_tokens, manifest.d_activation)

    out_dir = (
        Path(cfg.sae.ckpt_dir) / cfg.run_id / spec.slug / f"{arch}_w{width}_l0_{l0_target}"
    )
    train_sae(
        cfg=cfg.sae,
        cache_dir=cache_dir,
        d_in=manifest.d_activation,
        arch=arch,
        d_sae=width,
        l0_target=l0_target,
        out_dir=out_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
