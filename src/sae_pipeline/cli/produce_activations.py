"""Write fills of activations for SAE consumers on other GPUs or nodes.

Runs the model in vLLM with ``target.sites`` hooked and prefills one call of
training sequences from the configured stream at a time, writing each under
``--out`` as ``fill_<i>/`` with one file per site (see
``extraction.vllm.produce_fills``). With ``extraction.vllm.sequences_per_call:
auto`` each fill is as large as the free VRAM allows. Prints one JSON line per
fill.

With ``--consumers``, runs until those consumers (``cli.train_sae_streaming
--fills``) are done, deleting what they have read and keeping at most
``--max-buffer-gb`` of fills ahead of them. A rerun continues after the newest
fill; ``--start-state`` (a ``stream_state.pt`` of a streaming run) sets where
the first one starts.

Usage:
    python -m sae_pipeline.cli.produce_activations --config <config.yml> --out <dir> --fills 2
    python -m sae_pipeline.cli.produce_activations --config <config.yml> --out <dir> --consumers c0 c1 c2
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from sae_pipeline.cli._process import run_then_hard_exit
from sae_pipeline.config import PipelineCfg
from sae_pipeline.data.streaming import ResumableTokenStream
from sae_pipeline.extraction.fills import FillQueue
from sae_pipeline.extraction.vllm import produce_fills
from sae_pipeline.hooks.components import select_sites, site_requests
from sae_pipeline.model.loader import load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--fills", type=int, default=None, help="stop after this many (default: run with --consumers)")
    parser.add_argument("--consumers", nargs="+", default=None, help="run until these consumers are done")
    parser.add_argument("--max-buffer-gb", type=float, default=200.0, help="with --consumers: fills kept ahead")
    parser.add_argument("--start-state", type=Path, default=None, help="token stream state for the first fill")
    parser.add_argument("--stop-after-seconds", type=float, default=None)
    parser.add_argument("--threads", type=int, default=8, help="sites written at once")
    parser.add_argument("--sites", nargs="+", default=None, help="hook only these target.sites slugs")
    args = parser.parse_args()
    if args.fills is None and args.consumers is None:
        parser.error("give --fills, --consumers or both")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = PipelineCfg.from_yaml(args.config)
    targets = select_sites(cfg.target.sites or [], args.sites)
    specs = {slug: spec for slug, (spec, _site) in site_requests(targets).items()}
    token_stream = ResumableTokenStream(cfg.data, load_tokenizer(cfg.model))
    reports = produce_fills(
        cfg,
        specs,
        token_stream,
        out_dir=args.out,
        threads=args.threads,
        fills=args.fills,
        queue=None if args.consumers is None else FillQueue(args.out, args.consumers),
        max_buffer_bytes=int(args.max_buffer_gb * 1e9),
        start_state=args.start_state,
        stop_after_seconds=args.stop_after_seconds,
    )
    for report in reports:
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    run_then_hard_exit(main)
