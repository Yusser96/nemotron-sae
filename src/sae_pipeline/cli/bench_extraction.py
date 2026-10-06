"""Measure extraction throughput of one backend at several batch sizes.

Builds the backend once for ``target.sites`` (vLLM's scheduler budget sized for
the largest batch), then for each ``--batch-sequences`` value times
``--repeats`` extract() calls on fresh rows after one warm-up call. Rows come
from ``--token-ids`` (tiled) or are random ids. Prints one JSON line per batch
size: tokens/s is end to end, i.e. activations for every site delivered on
``--capture-device``.

Usage:
    python -m sae_pipeline.cli.bench_extraction --config <cfg.yml> --backend vllm \\
        --batch-sequences 1 4 8 16 32 --repeats 3 --out bench.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import torch
from safetensors.torch import load_file

from sae_pipeline.cli._process import run_then_hard_exit
from sae_pipeline.config import PipelineCfg
from sae_pipeline.extraction.base import ActivationExtractor
from sae_pipeline.extraction.build import build_extractor


def bench_extraction(
    extractor: ActivationExtractor, rows: torch.Tensor, batch_sequences: list[int], *, repeats: int
) -> list[dict]:
    """Time ``repeats`` extract() calls per batch size; ``rows`` must hold enough distinct rows."""
    results = []
    for batch in batch_sequences:
        calls = [rows[i * batch : (i + 1) * batch] for i in range(repeats + 1)]
        extractor.extract(calls[0])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        for call in calls[1:]:
            extractor.extract(call)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        seconds = (time.perf_counter() - started) / repeats
        tokens = batch * rows.shape[1]
        results.append(
            {"batch_sequences": batch, "tokens_per_call": tokens, "seconds_per_call": seconds, "tokens_per_s": tokens / seconds}
        )
    return results


def _rows(cfg: PipelineCfg, token_ids_path: str | None, count: int) -> torch.Tensor:
    if token_ids_path is None:
        generator = torch.Generator().manual_seed(0)
        return torch.randint(1_000, 100_000, (count, cfg.data.seq_len), generator=generator)
    source = load_file(token_ids_path)["token_ids"]
    return source.repeat((count + source.shape[0] - 1) // source.shape[0], 1)[:count]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--backend", choices=["hf_hooks", "vllm"], required=True)
    parser.add_argument("--batch-sequences", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--token-ids", default=None, help="safetensors file with token_ids rows to tile")
    parser.add_argument("--device", default="cuda:0", help="model device for hf_hooks")
    parser.add_argument("--capture-device", default="cuda:0")
    parser.add_argument("--out", default=None, help="append JSON lines here")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = PipelineCfg.from_yaml(args.config).with_overrides(
        **{"extraction.vllm.sequences_per_call": max(args.batch_sequences)}
    )
    extractor, _ = build_extractor(cfg, args.backend, device=args.device, capture_device=args.capture_device)
    rows = _rows(cfg, args.token_ids, max(args.batch_sequences) * (args.repeats + 1))
    try:
        results = bench_extraction(extractor, rows, args.batch_sequences, repeats=args.repeats)
    finally:
        extractor.close()
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    lines = [json.dumps({"backend": args.backend, "gpu": gpu, "sites": len(cfg.target.sites or []), **r}) for r in results]
    print("\n".join(lines))
    if args.out:
        with open(args.out, "a") as handle:
            handle.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    run_then_hard_exit(main)
