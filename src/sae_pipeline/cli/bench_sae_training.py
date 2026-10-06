"""Measure how fast one GPU trains a set of SAEs, and how much memory is left over.

Runs the streaming trainer's step (``sae.train_streaming.streaming_step``) for
``--sites`` SAEs on synthetic activations already on the device, so the
number is the consumer's capacity independent of any extraction backend.
Reports tokens/s (per site), peak memory, and how many tokens of
``--sites`` x ``d_in`` bf16 activations would fit in the remaining memory --
the upper bound for an on-GPU activation buffer.

Usage:
    python -m sae_pipeline.cli.bench_sae_training --config <cfg.yml> --sites 14 --d-in 2688 --steps 200
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import torch

from sae_pipeline.config import PipelineCfg
from sae_pipeline.sae.train import build_sae_and_optimizer
from sae_pipeline.sae.train_streaming import StreamingSite, streaming_step


def _site(index: int, cfg: PipelineCfg, d_in: int, device: str, out_dir: Path) -> StreamingSite:
    sae_cfg = cfg.sae
    d_sae = sae_cfg.d_sae if isinstance(sae_cfg.d_sae, int) else sae_cfg.d_sae[0]
    l0_target = sae_cfg.l0_target if isinstance(sae_cfg.l0_target, int) else sae_cfg.l0_target[0]
    sae, optim = build_sae_and_optimizer(sae_cfg, arch="jumprelu", d_in=d_in, d_sae=d_sae, device=device)
    return StreamingSite(
        slug=f"site{index}",
        cfg=sae_cfg,
        sae=sae,
        optim=optim,
        arch="jumprelu",
        l0_target=l0_target,
        normalization_scale=1.0,
        out_dir=out_dir,
        validation_readers={},
        validation_cache_signature={},
        probe_tokens=0,
        ever_fired=torch.zeros(d_sae, dtype=torch.bool, device=device),
        feature_frequency=torch.zeros(d_sae, dtype=torch.float32, device=device),
        log_f=open(os.devnull, "w"),
        validation_f=open(os.devnull, "w"),
    )


def bench_sae_training(cfg: PipelineCfg, *, sites: int, d_in: int, steps: int, warmup: int, device: str) -> dict:
    batch_size = cfg.sae.batch_size
    with tempfile.TemporaryDirectory() as out_dir:
        trained = [_site(index, cfg, d_in, device, Path(out_dir)) for index in range(sites)]
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sites", type=int, default=14)
    parser.add_argument("--d-in", type=int, default=2688)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=None, help="append the JSON result as one line here")
    args = parser.parse_args()

    report = bench_sae_training(
        PipelineCfg.from_yaml(args.config),
        sites=args.sites,
        d_in=args.d_in,
        steps=args.steps,
        warmup=args.warmup,
        device=args.device,
    )
    line = json.dumps(report)
    print(line)
    if args.out:
        with open(args.out, "a") as handle:
            handle.write(line + "\n")


if __name__ == "__main__":
    main()
