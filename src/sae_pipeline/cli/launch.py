"""Materialize the prod sweep over (layer × component × arch × width × l0) into jobs.

Each job is just a `python -m sae_pipeline.cli.{cache_activations,train_sae,evaluate}` call.
The `--executor` flag picks how to run them: dry (print), local (subprocess pool sized
by CUDA_VISIBLE_DEVICES), or slurm (later).
"""

from __future__ import annotations

import argparse
import itertools
import logging
import os
import shlex
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from sae_pipeline.config import PipelineCfg

log = logging.getLogger(__name__)


def _aslist(x):
    return x if isinstance(x, list) else [x]


def materialize_jobs(cfg: PipelineCfg, config_path: str) -> list[list[str]]:
    layers = _aslist(cfg.target.layer)
    components = _aslist(cfg.target.component)
    arches = _aslist(cfg.sae.arch)
    widths = _aslist(cfg.sae.d_sae)
    l0s = _aslist(cfg.sae.l0_target)

    jobs: list[list[str]] = []

    # Cache jobs: one per (layer, component) — independent of (arch, width, l0).
    for L, C in itertools.product(layers, components):
        jobs.append([
            "python", "-m", "sae_pipeline.cli.cache_activations",
            "--config", config_path,
            "--layer", str(L),
            "--component", str(C),
        ])

    # Train + eval jobs: full cartesian product over the SAE knobs.
    for L, C, arch, w, l0 in itertools.product(layers, components, arches, widths, l0s):
        jobs.append([
            "python", "-m", "sae_pipeline.cli.train_sae",
            "--config", config_path,
            "--layer", str(L),
            "--component", str(C),
            "--arch", str(arch),
            "--width", str(w),
            "--l0", str(l0),
        ])
        jobs.append([
            "python", "-m", "sae_pipeline.cli.evaluate",
            "--config", config_path,
            "--layer", str(L),
            "--component", str(C),
            "--arch", str(arch),
            "--width", str(w),
            "--l0", str(l0),
        ])
    return jobs


def _gpus_from_env() -> list[str]:
    v = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    return [g for g in v.split(",") if g] or ["0"]


def run_local(jobs: list[list[str]], gpus: list[str]) -> int:
    """Run jobs in parallel; pin each to one GPU via CUDA_VISIBLE_DEVICES."""
    n_workers = max(1, len(gpus))
    failures = 0

    def _run(job_and_gpu):
        cmd, gpu = job_and_gpu
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        log.info("[GPU %s] %s", gpu, " ".join(map(shlex.quote, cmd)))
        return subprocess.run(cmd, env=env, check=False).returncode

    # Round-robin GPU assignment.
    tagged = [(cmd, gpus[i % n_workers]) for i, cmd in enumerate(jobs)]
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(_run, t) for t in tagged]
        for fut in as_completed(futures):
            if fut.result() != 0:
                failures += 1
    return failures


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--executor", choices=["dry", "local"], default="dry")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = PipelineCfg.from_yaml(args.config)
    jobs = materialize_jobs(cfg, args.config)
    log.info("Materialized %d jobs from %s", len(jobs), args.config)

    if args.executor == "dry":
        for cmd in jobs:
            print(" ".join(map(shlex.quote, cmd)))
        return
    if args.executor == "local":
        gpus = _gpus_from_env()
        nfail = run_local(jobs, gpus)
        if nfail:
            log.error("%d job(s) failed", nfail)
            raise SystemExit(1)


if __name__ == "__main__":
    main()
