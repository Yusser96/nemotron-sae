"""Phase A: cache activations for one (layer, component) to safetensors shards.

Usage:
    python -m sae_pipeline.cli.cache_activations \
        --config configs/dev.yaml \
        --layer 25 --component resid_post
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from tqdm import tqdm

from sae_pipeline.cache.manifest import (
    CacheManifest,
    manifest_path_for,
    shards_dir_for,
)
from sae_pipeline.cache.writer import ShardWriter
from sae_pipeline.config import PipelineCfg
from sae_pipeline.data.streaming import make_token_loader
from sae_pipeline.hooks.components import ComponentSpec, load_topology, resolve
from sae_pipeline.hooks.extractor import capture
from sae_pipeline.model.loader import load_model_and_tokenizer
from sae_pipeline.model.topology import dump_topology, enumerate_hooks

log = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--component", required=True, help="e.g. resid_post, moe_out, expert.42")
    p.add_argument("--max-batches", type=int, default=None,
                   help="Cap the number of forward passes (debugging).")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = PipelineCfg.from_yaml(args.config)
    spec = ComponentSpec.parse(args.layer, args.component)

    model, tokenizer = load_model_and_tokenizer(cfg.model)
    sites = enumerate_hooks(model)
    topology_path = Path(cfg.cache.cache_dir) / cfg.run_id / "model_topology.json"
    dump_topology(sites, topology_path)
    site = resolve(spec, sites)
    log.info("Resolved %s -> %s", spec, site)

    # The activation dimension depends on the component:
    #   - resid_*, moe_out, mamba_out, attn_out_prelinear: typically d_model
    #   - expert / shared_expert: also d_model (they output back into the residual)
    # We probe shape on the first hook fire and only then construct the writer.

    out_dir = shards_dir_for(cfg.cache.cache_dir, cfg.run_id, spec.slug)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Caching to %s", out_dir)

    manifest = CacheManifest(
        run_id=cfg.run_id,
        model=cfg.model.name,
        dtype=cfg.model.dtype,
        layer=spec.layer,
        component=args.component,
        d_activation=-1,   # patched after first capture
        shuffle_seed=cfg.cache.shuffle_seed,
    )

    # Pick a microbatch dimension that matches tokens_per_fwd / seq_len.
    micro_batch = max(1, cfg.cache.tokens_per_fwd // cfg.data.seq_len)
    loader = make_token_loader(cfg.data, tokenizer, batch_size=micro_batch)

    writer: ShardWriter | None = None
    n_seen = 0
    device = next(model.parameters()).device

    for i, batch in enumerate(tqdm(loader, desc="forward")):
        if args.max_batches is not None and i >= args.max_batches:
            break
        batch = batch.to(device)
        with capture(model, spec, site) as buf:
            with torch.no_grad():
                model(batch)
            captured = buf.consume()

        if captured.numel() == 0:
            log.warning("Empty capture at step %d (component=%s)", i, args.component)
            continue
        if writer is None:
            d = captured.shape[-1]
            manifest.d_activation = d
            writer = ShardWriter(
                out_dir=out_dir,
                d_activation=d,
                shard_size_bytes=cfg.cache.shard_size_bytes,
                dtype=torch.bfloat16 if cfg.model.dtype == "bfloat16" else torch.float32,
                shuffle_seed=cfg.cache.shuffle_seed,
                manifest=manifest,
            )
        writer.add(captured)
        n_seen += captured.shape[0]

    if writer is None:
        log.error("Captured zero activations — check the component/layer.")
        raise SystemExit(2)

    final_manifest = writer.close()
    log.info(
        "Cache done: %d shards, %d tokens at %s",
        final_manifest.n_shards, final_manifest.total_tokens,
        manifest_path_for(cfg.cache.cache_dir, cfg.run_id, spec.slug),
    )


if __name__ == "__main__":
    main()
