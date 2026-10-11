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
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

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
    """Run cache, train and evaluation phases with exclusive GPU leases.

    A failed phase prevents its dependent phases from starting. Threads only
    supervise subprocesses; model and training work runs in those processes.
    """
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError("Provide at least one GPU, with no duplicate IDs")
    available: Queue[str] = Queue()
    for gpu in gpus:
        available.put(gpu)
    failures = 0

    def _run(cmd):
        gpu = available.get()
        try:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            log.info("[GPU %s] %s", gpu, " ".join(map(shlex.quote, cmd)))
            return subprocess.run(cmd, env=env, check=False).returncode
        finally:
            available.put(gpu)

    modules = ("cache_activations", "train_sae", "evaluate")
    def phase(cmd):
        module = cmd[2].rsplit(".", 1)[-1] if len(cmd) > 2 and cmd[1] == "-m" else ""
        return modules.index(module) if module in modules else 0

    with ThreadPoolExecutor(max_workers=len(gpus)) as ex:
        for index in range(len(modules)):
            futures = [ex.submit(_run, cmd) for cmd in jobs if phase(cmd) == index]
            for fut in as_completed(futures):
                if fut.result() != 0:
                    failures += 1
            if failures:
                break
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
