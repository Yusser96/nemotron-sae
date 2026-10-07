"""Write fills of activations for SAE consumers on other GPUs or nodes.

Runs the model in vLLM with ``target.sites`` hooked and, ``--fills`` times,
prefills one call of training sequences from the configured stream and writes
it under ``--out`` as ``fill_<i>/`` with one file per site (see
``extraction.vllm.produce_fills``). With ``extraction.vllm.sequences_per_call:
auto`` each fill is as large as the free VRAM allows. Prints one JSON line per
fill.

Usage:
    python -m sae_pipeline.cli.produce_activations --config <config.yml> --out <dir> --fills 2
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from sae_pipeline.cli._process import run_then_hard_exit
from sae_pipeline.config import PipelineCfg
from sae_pipeline.data.streaming import ResumableTokenStream
from sae_pipeline.extraction.vllm import produce_fills
from sae_pipeline.hooks.components import site_requests
from sae_pipeline.model.loader import load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--fills", type=int, default=2)
    parser.add_argument("--threads", type=int, default=8, help="sites written at once")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = PipelineCfg.from_yaml(args.config)
    specs = {slug: spec for slug, (spec, _site) in site_requests(cfg.target.sites or []).items()}
    token_stream = ResumableTokenStream(cfg.data, load_tokenizer(cfg.model))
    reports = produce_fills(cfg, specs, token_stream, out_dir=args.out, fills=args.fills, threads=args.threads)
    for report in reports:
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    run_then_hard_exit(main)
