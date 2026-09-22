"""Phase A: cache activations for one (layer, component) to safetensors shards.

Usage:
    python -m sae_pipeline.cli.cache_activations \
        --config configs/dev.yaml \
        --layer 25 --component resid_post
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from tqdm import tqdm

from sae_pipeline.cache.manifest import CacheManifest
from sae_pipeline.cache.writer import CacheBudget, ShardWriter
from sae_pipeline.config import PipelineCfg
from sae_pipeline.data.streaming import make_token_loader
from sae_pipeline.hooks.components import ComponentSpec, resolve
from sae_pipeline.hooks.extractor import capture_many
from sae_pipeline.model.loader import load_model_and_tokenizer
from sae_pipeline.model.topology import HookSite, dump_topology, enumerate_hooks

log = logging.getLogger(__name__)


def _repair_misaligned_manifests(
    *,
    cache_root: Path,
    base_dir: Path,
    manifests: dict[str, CacheManifest],
    token_multiple: int,
) -> int:
    """Roll back interrupted multi-site writes to their common valid prefix.

    Each site writes its manifest independently.  If Slurm terminates the job
    while the sites are being flushed, a few manifests can contain one or more
    extra complete shards.  The common prefix is valid because all sites were
    captured by the same forwards.  Extra files are moved outside the cache
    root for recovery rather than removed.
    """
    counts = [manifest.total_tokens for manifest in manifests.values()]
    common_tokens = min(counts)
    if common_tokens % token_multiple:
        raise RuntimeError(
            "Misaligned cache manifests do not share a tokens_per_fwd boundary: "
            f"counts={counts}, tokens_per_fwd={token_multiple}"
        )

    recovery_root = cache_root.parent / (
        f".{cache_root.name}.recovery-{os.getpid()}-{time.time_ns()}"
    )
    for slug, manifest in manifests.items():
        site_dir = base_dir / slug
        keep: list[str] = []
        kept_tokens = 0
        for name in manifest.shard_paths:
            shard_path = site_dir / name
            if not shard_path.exists():
                raise RuntimeError(f"Manifest references missing shard {shard_path}")
            with safe_open(str(shard_path), framework="pt", device="cpu") as shard:
                shard_tokens = int(shard.get_tensor("x").shape[0])
            if kept_tokens + shard_tokens > common_tokens:
                if kept_tokens != common_tokens:
                    raise RuntimeError(
                        f"Cannot safely align {shard_path}: shard crosses common "
                        f"boundary at {common_tokens} tokens"
                    )
                break
            keep.append(name)
            kept_tokens += shard_tokens

        if kept_tokens != common_tokens:
            raise RuntimeError(
                f"Could not align {slug} to {common_tokens} tokens; "
                f"its shard prefix contains {kept_tokens}"
            )

        keep_set = set(keep)
        for shard_path in site_dir.glob("shard_*.safetensors"):
            if shard_path.name in keep_set:
                continue
            destination = recovery_root / slug / shard_path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(shard_path, destination)

        manifest.shard_paths = keep
        manifest.n_shards = len(keep)
        manifest.total_tokens = kept_tokens
        manifest.complete = False
        manifest.write(site_dir / "manifest.json")

    log.warning(
        "Recovered interrupted multi-site cache to common prefix %d tokens; "
        "extra shards moved to %s",
        common_tokens,
        recovery_root,
    )
    return common_tokens


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--layer", type=int)
    p.add_argument("--component", help="e.g. resid_post, moe_out, expert.42")
    p.add_argument("--all-targets", action="store_true",
                   help="Capture every exact hook listed in target.sites in one forward pass.")
    p.add_argument("--partition", choices=["train", "validation"], default="train")
    p.add_argument("--language", help="Language label for a validation cache, e.g. de or en.")
    p.add_argument("--max-batches", type=int, default=None,
                   help="Cap the number of forward passes (debugging).")
    args = p.parse_args()

    if args.all_targets == (args.layer is not None or args.component is not None):
        p.error("Use either --all-targets or both --layer and --component")
    if (args.layer is None) != (args.component is None):
        p.error("--layer and --component must be supplied together")
    if args.partition == "validation" and not args.language:
        p.error("--language is required for a validation cache")
    if args.partition == "train" and args.language:
        p.error("--language is only valid with --partition validation")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Slurm sends TERM shortly before a one-hour allocation expires.  Defer
    # stopping until the current full forward has been written for every hook,
    # preserving equal, forward-aligned manifests across the whole SAE suite.
    stop_requested = False

    def request_stop(signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True
        log.warning("Received signal %s; finishing the current forward before checkpointing the cache", signum)

    signal.signal(signal.SIGTERM, request_stop)

    cfg = PipelineCfg.from_yaml(args.config)
    if cfg.cache.tokens_per_fwd <= 0:
        p.error("cache.tokens_per_fwd must be positive")
    if cfg.cache.tokens_per_fwd % cfg.data.seq_len:
        p.error("cache.tokens_per_fwd must be divisible by data.seq_len")

    model, tokenizer = load_model_and_tokenizer(cfg.model)
    sites = enumerate_hooks(model)
    topology_path = Path(cfg.cache.cache_dir) / cfg.run_id / "model_topology.json"
    dump_topology(sites, topology_path)
    requests: dict[str, tuple[ComponentSpec, HookSite]] = {}
    components: dict[str, str] = {}
    layers: dict[str, int] = {}
    if args.all_targets:
        if not cfg.target.sites:
            p.error("--all-targets requires target.sites")
        for target in cfg.target.sites:
            spec = ComponentSpec.parse(target.layer, target.component)
            # Released SAE metadata is authoritative.  It avoids guessing the
            # custom module type from a path such as `.mixer`.
            site = HookSite(target.component, target.layer, target.hook_name)
            requests[target.slug] = (spec, site)
            components[target.slug] = target.component
            layers[target.slug] = target.layer
    else:
        assert args.layer is not None and args.component is not None
        spec = ComponentSpec.parse(args.layer, args.component)
        requests[spec.slug] = (spec, resolve(spec, sites))
        components[spec.slug] = args.component
        layers[spec.slug] = args.layer

    cache_root = Path(cfg.cache.cache_dir) / cfg.run_id
    base_dir = cache_root if args.partition == "train" else cache_root / "validation" / args.language
    fingerprint = hashlib.sha256(json.dumps(cfg.data.model_dump(), sort_keys=True).encode()).hexdigest()
    manifests: dict[str, CacheManifest] = {}
    writers: dict[str, ShardWriter] = {}
    target_tokens = (
        cfg.data.total_tokens
        if args.partition == "train"
        else cfg.data.validation_tokens_per_language
    )
    for slug in requests:
        out_dir = base_dir / slug
        out_dir.mkdir(parents=True, exist_ok=True)
        expected = CacheManifest(
            run_id=cfg.run_id,
            model=cfg.model.name,
            dtype=cfg.model.dtype,
            layer=layers[slug],
            component=components[slug],
            d_activation=-1,
            shuffle_seed=cfg.cache.shuffle_seed,
            dataset_fingerprint=fingerprint,
            partition=args.partition,
            language=args.language,
        )
        existing_path = out_dir / "manifest.json"
        if existing_path.exists():
            existing = CacheManifest.read(existing_path)
            for field in ("run_id", "model", "layer", "component", "dataset_fingerprint", "partition", "language"):
                if getattr(existing, field) != getattr(expected, field):
                    raise RuntimeError(f"Existing cache {existing_path} has incompatible {field}")
            if target_tokens is not None and existing.total_tokens > target_tokens:
                raise RuntimeError(f"Existing cache {existing_path} exceeds its configured token budget")
            manifests[slug] = existing
        else:
            manifests[slug] = expected
    log.info("Caching %d sites to %s", len(requests), base_dir)

    # Pick a microbatch dimension that matches tokens_per_fwd / seq_len.
    micro_batch = max(1, cfg.cache.tokens_per_fwd // cfg.data.seq_len)
    languages = {args.language} if args.language else None
    loader = make_token_loader(
        cfg.data, tokenizer, batch_size=micro_batch,
        partition=args.partition, languages=languages,
    )

    existing_counts = {manifest.total_tokens for manifest in manifests.values()}
    if len(existing_counts) != 1:
        if any(manifest.complete for manifest in manifests.values()):
            raise RuntimeError(
                "Cannot repair cache manifests when only some sites are marked complete"
            )
        _repair_misaligned_manifests(
            cache_root=cache_root,
            base_dir=base_dir,
            manifests=manifests,
            token_multiple=cfg.cache.tokens_per_fwd,
        )
        existing_counts = {manifest.total_tokens for manifest in manifests.values()}
    budget = CacheBudget(cache_root, cfg.cache.max_total_bytes)
    if len(existing_counts) != 1:
        raise RuntimeError("All sites in a multi-site cache must resume from the same token count")
    already_cached = existing_counts.pop()
    complete_states = {manifest.complete for manifest in manifests.values()}
    if len(complete_states) != 1:
        raise RuntimeError("All sites in a multi-site cache must have the same completion state")
    if complete_states == {True}:
        if target_tokens is None or already_cached == target_tokens:
            log.info("All requested caches are already complete at %d tokens", already_cached)
            return
        raise RuntimeError("Completed cache does not match its configured token budget")
    if target_tokens is not None and already_cached == target_tokens:
        for slug, manifest in manifests.items():
            manifest.complete = True
            manifest.write(base_dir / slug / "manifest.json")
        log.info("Marked existing caches complete at %d tokens", target_tokens)
        return
    if already_cached % cfg.cache.tokens_per_fwd:
        raise RuntimeError("Existing cache length must align to tokens_per_fwd for deterministic resume")
    resume_batches = already_cached // cfg.cache.tokens_per_fwd
    log.info("Resuming after %d cached tokens (%d forward passes)", already_cached, resume_batches)
    device = next(model.parameters()).device
    stop_after_seconds = float(os.environ.get("CACHE_STOP_AFTER_SECONDS", "0"))
    if stop_after_seconds < 0:
        raise RuntimeError("CACHE_STOP_AFTER_SECONDS must be non-negative")
    loop_started = time.monotonic()

    for i, batch in enumerate(tqdm(loader, desc="forward")):
        if args.max_batches is not None and i >= args.max_batches:
            break
        if i < resume_batches:
            continue
        if stop_after_seconds and time.monotonic() - loop_started >= stop_after_seconds:
            stop_requested = True
            log.info(
                "Stopping at the configured self-managed time boundary after %.0fs",
                stop_after_seconds,
            )
            break
        batch = batch.to(device)
        with capture_many(model, requests) as buffers:
            with torch.no_grad():
                model(batch)
            captured = {slug: buffer.consume() for slug, buffer in buffers.items()}

        for slug, x in captured.items():
            if x.numel() == 0:
                raise RuntimeError(f"Empty capture at step {i} for {slug}")
            if slug not in writers:
                if manifests[slug].d_activation not in {-1, x.shape[-1]}:
                    raise RuntimeError(
                        f"Cached {slug} has d={manifests[slug].d_activation}, new capture has d={x.shape[-1]}"
                    )
                manifests[slug].d_activation = x.shape[-1]
                writers[slug] = ShardWriter(
                    out_dir=base_dir / slug,
                    d_activation=x.shape[-1],
                    shard_size_bytes=cfg.cache.shard_size_bytes,
                    dtype=torch.bfloat16 if cfg.model.dtype == "bfloat16" else torch.float32,
                    shuffle_seed=cfg.cache.shuffle_seed,
                    manifest=manifests[slug],
                    cache_budget=budget,
                    flush_token_multiple=cfg.cache.tokens_per_fwd,
                )
            writers[slug].add(x)

        if stop_requested:
            log.info("Stopping cleanly at the next committed cache boundary")
            break

    if len(writers) != len(requests):
        if stop_requested:
            log.info("Stopped before a complete forward was captured; cache remains resumable")
            return
        if target_tokens is None and already_cached > 0 and args.max_batches is None:
            for slug, manifest in manifests.items():
                manifest.complete = True
                manifest.write(base_dir / slug / "manifest.json")
            log.info("Existing document-budget cache is complete at %d tokens", already_cached)
            return
        missing = sorted(set(requests) - set(writers))
        raise RuntimeError(f"Captured zero activations for: {missing}")
    for slug, writer in writers.items():
        final_manifest = writer.close()
        if (
            target_tokens is not None
            and args.max_batches is None
            and not stop_requested
            and final_manifest.total_tokens != target_tokens
        ):
            raise RuntimeError(
                f"Cache {slug} ended at {final_manifest.total_tokens} tokens; "
                f"expected {target_tokens}"
            )
        if args.max_batches is None and not stop_requested:
            final_manifest.complete = True
            final_manifest.write(base_dir / slug / "manifest.json")
        log.info("Cache done: %s (%d shards, %d tokens)", slug,
                 final_manifest.n_shards, final_manifest.total_tokens)


if __name__ == "__main__":
    main()
    # Triton may leave an autotuner worker thread alive after the CUDA work has
    # completed.  On this model/runtime combination, normal interpreter
    # finalisation can then abort with PyGILState_Release even though every
    # cache shard and manifest has been committed.  Flush the CLI output and
    # bypass that shutdown path only after main() has returned successfully.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
