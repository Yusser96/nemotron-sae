"""Measure how fast one GPU trains a set of SAEs, and how much memory is left over.

Runs the streaming trainer's step (``sae.train_streaming.streaming_step``) for
``--sites`` SAEs on synthetic activations already on the device, so the
number is the consumer's capacity independent of any extraction backend.
Reports tokens/s (per site), peak memory, and how many tokens of
``--sites`` x ``d_in`` bf16 activations would fit in the remaining memory --
the upper bound for an on-GPU activation buffer.

With ``--cache``, trains instead on the fills ``cli.produce_activations``
wrote (all their sites, or ``--slugs``), one site file after another, once
per ``--modes`` entry: ``stream`` loads the next file on a thread while
training, as a consumer reading a shared filesystem would; ``memory`` loads
each file into host memory first and times only the training. One JSON line
per mode.

Usage:
    python -m sae_pipeline.cli.bench_sae_training --config <cfg.yml> --sites 14 --d-in 2688 --steps 200
    python -m sae_pipeline.cli.bench_sae_training --config <cfg.yml> --cache <dir> --modes stream memory
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from sae_pipeline.config import PipelineCfg
from sae_pipeline.sae.train import build_sae_and_optimizer
from sae_pipeline.sae.train_streaming import StreamingSite, streaming_step


def _site(slug: str, cfg: PipelineCfg, d_in: int, device: str, out_dir: Path, scale: float = 1.0) -> StreamingSite:
    sae_cfg = cfg.sae
    d_sae = sae_cfg.d_sae if isinstance(sae_cfg.d_sae, int) else sae_cfg.d_sae[0]
    l0_target = sae_cfg.l0_target if isinstance(sae_cfg.l0_target, int) else sae_cfg.l0_target[0]
    sae, optim = build_sae_and_optimizer(sae_cfg, arch="jumprelu", d_in=d_in, d_sae=d_sae, device=device)
    return StreamingSite(
        slug=slug,
        cfg=sae_cfg,
        sae=sae,
        optim=optim,
        arch="jumprelu",
        l0_target=l0_target,
        normalization_scale=scale,
        out_dir=out_dir,
        validation_readers={},
        validation_cache_signature={},
        probe_tokens=0,
        ever_fired=torch.zeros(d_sae, dtype=torch.bool, device=device),
        feature_frequency=torch.zeros(d_sae, dtype=torch.float32, device=device),
        inverse_simpson_usage=torch.zeros(d_sae, dtype=torch.float32, device=device),
        tokens_since_fired=torch.zeros(d_sae, dtype=torch.int64, device=device),
        log_f=open(os.devnull, "w"),
        validation_f=open(os.devnull, "w"),
    )


def bench_sae_training(cfg: PipelineCfg, *, sites: int, d_in: int, steps: int, warmup: int, device: str) -> dict:
    batch_size = cfg.sae.batch_size
    with tempfile.TemporaryDirectory() as out_dir:
        trained = [_site(f"site{index}", cfg, d_in, device, Path(out_dir)) for index in range(sites)]
        # Unit-norm-ish synthetic activations in bf16, as captured activations arrive.
        batches = torch.randn(4, batch_size, d_in, device=device, dtype=torch.bfloat16) / d_in**0.5
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)
        started = None
        for step in range(1, warmup + steps + 1):
            if step == warmup + 1:
                if device.startswith("cuda"):
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
            for site in trained:
                streaming_step(site, batches[step % 4], step)
                site.step = step
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started
        for site in trained:
            site.close()
    report = {
        "device": torch.cuda.get_device_name(device) if device.startswith("cuda") else device,
        "sites": sites,
        "d_in": d_in,
        "batch_size": batch_size,
        "steps": steps,
        "steps_per_s": steps / seconds,
        "tokens_per_s_per_site": steps * batch_size / seconds,
    }
    if device.startswith("cuda"):
        free, total = torch.cuda.mem_get_info(device)
        peak = torch.cuda.max_memory_allocated(device)
        report.update(
            peak_allocated_gib=peak / 2**30,
            total_gib=total / 2**30,
            free_after_gib=free / 2**30,
            # Every buffered token holds one bf16 vector per site.
            buffer_tokens_that_fit=int((total - peak) // (sites * d_in * 2)),
        )
    return report


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def _load(path: Path) -> torch.Tensor:
    return load_file(path)["activations"]


def _fill_site(path: Path, cfg: PipelineCfg, device: str, out_dir: Path) -> StreamingSite:
    """An SAE for this site file's width, scaled to unit mean norm on its first batch."""
    with safe_open(path, "pt") as handle:
        sample = handle.get_slice("activations")[: cfg.sae.batch_size].float()
    return _site(path.stem, cfg, sample.shape[1], device, out_dir, scale=1.0 / sample.norm(dim=1).mean().item())


def bench_fills(cfg: PipelineCfg, fill_dirs: list[Path], slugs: list[str], *, modes: list[str], device: str) -> list[dict]:
    """Train one SAE per slug over every fill, once per mode (see the module docstring)."""
    batch_size = cfg.sae.batch_size
    paths = [fill / f"{slug}.safetensors" for fill in fill_dirs for slug in slugs]
    reports = []
    with tempfile.TemporaryDirectory() as out_dir, ThreadPoolExecutor(max_workers=1) as loader:
        sites = {slug: _fill_site(fill_dirs[0] / f"{slug}.safetensors", cfg, device, Path(out_dir)) for slug in slugs}
        for mode in modes:
            if device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(device)
            steps, nbytes, waiting, loading = 0, 0, 0.0, 0.0
            upcoming = loader.submit(_load, paths[0]) if mode == "stream" else None
            _sync(device)
            started = time.perf_counter()
            for index, path in enumerate(paths):
                if mode == "stream":
                    wait_started = time.perf_counter()
                    rows = upcoming.result()
                    waiting += time.perf_counter() - wait_started
                    if index + 1 < len(paths):
                        upcoming = loader.submit(_load, paths[index + 1])
                else:
                    _sync(device)
                    load_started = time.perf_counter()
                    rows = _load(path)
                    loading += time.perf_counter() - load_started
                nbytes += rows.nbytes
                site = sites[path.stem]
                for batch in rows.split(batch_size):
                    if batch.shape[0] < batch_size:
                        break
                    site.step += 1
                    streaming_step(site, batch.to(device), site.step)
                    steps += 1
            _sync(device)
            seconds = time.perf_counter() - started - loading
            report = {
                "mode": mode,
                "device": torch.cuda.get_device_name(device) if device.startswith("cuda") else device,
                "sites": len(slugs),
                "fills": len(fill_dirs),
                "site_steps": steps,
                "seconds": seconds,
                # One step of every site: the rate the whole consumer trains at.
                "steps_per_s": steps / len(slugs) / seconds,
                "tokens_per_s": steps / len(slugs) * batch_size / seconds,
                "waiting_s": waiting,
                "load_gb_per_s": nbytes / loading / 1e9 if loading else None,
            }
            if device.startswith("cuda"):
                report["peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
            reports.append(report)
        for site in sites.values():
            site.close()
    return reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sites", type=int, default=14)
    parser.add_argument("--d-in", type=int, default=2688)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache", type=Path, default=None, help="train on these fills instead of synthetic batches")
    parser.add_argument("--slugs", nargs="+", default=None, help="with --cache: the sites to train (default: all)")
    parser.add_argument("--modes", nargs="+", choices=["stream", "memory"], default=["stream", "memory"])
    parser.add_argument("--out", default=None, help="append the JSON results as lines here")
    args = parser.parse_args()

    cfg = PipelineCfg.from_yaml(args.config)
    if args.cache is None:
        reports = [
            bench_sae_training(
                cfg, sites=args.sites, d_in=args.d_in, steps=args.steps, warmup=args.warmup, device=args.device
            )
        ]
    else:
        fill_dirs = sorted(args.cache.glob("fill_*"))
        slugs = args.slugs or sorted(p.stem for p in fill_dirs[0].glob("*.safetensors") if p.stem != "token_ids")
        reports = bench_fills(cfg, fill_dirs, slugs, modes=args.modes, device=args.device)
    lines = "\n".join(json.dumps(report) for report in reports)
    print(lines)
    if args.out:
        with open(args.out, "a") as handle:
            handle.write(lines + "\n")


if __name__ == "__main__":
    main()
