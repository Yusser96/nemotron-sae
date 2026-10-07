"""Run the model in vLLM and stream its activations to a trainer process.

The producer half of ``extraction.backend: vllm_remote`` (see
``extraction.remote``): start it on the same node as
``cli.train_sae_streaming``, with the same config. vLLM takes the first
visible GPU, so ``stream.sae_device`` should name another one. It hooks
``target.sites``, answers every token batch the trainer sends and exits when
the trainer stops.

Usage:
    python -m sae_pipeline.cli.serve_activations --config <config.yml>
"""

from __future__ import annotations

import argparse
import logging

from sae_pipeline.cli._process import run_then_hard_exit
from sae_pipeline.config import PipelineCfg
from sae_pipeline.extraction.vllm import serve_activations
from sae_pipeline.hooks.components import site_requests

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = PipelineCfg.from_yaml(args.config)
    specs = {slug: spec for slug, (spec, _site) in site_requests(cfg.target.sites or []).items()}
    remote = cfg.extraction.remote
    served = serve_activations(cfg, specs, host=remote.host, port=remote.port)
    log.info("Served %d token batches", served)


if __name__ == "__main__":
    run_then_hard_exit(main)
