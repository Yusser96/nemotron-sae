"""Dump activations for a fixed token set with one extraction backend.

Used to compare backends on identical inputs (see ``cli.compare_activations``).
The first run samples validation-partition sequences from ``data.sources`` and
writes them to ``--token-ids``; later runs (e.g. with another backend, possibly
in another environment) reuse that file so every backend sees the same tokens.

Usage:
    python -m sae_pipeline.cli.dump_activations --config <cfg.yml> --backend hf_hooks \\
        --token-ids tokens.safetensors --n-sequences 32 --out hf.safetensors
    python -m sae_pipeline.cli.dump_activations --config <cfg.yml> --backend vllm \\
        --token-ids tokens.safetensors --out vllm.safetensors
"""

from __future__ import annotations

import argparse
import itertools
import logging
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from sae_pipeline.cli._process import run_then_hard_exit
from sae_pipeline.config import PipelineCfg
from sae_pipeline.extraction.base import ActivationExtractor
from sae_pipeline.hooks.components import ComponentSpec, site_requests

log = logging.getLogger(__name__)


def sample_validation_token_ids(cfg: PipelineCfg, tokenizer, n_sequences: int) -> torch.Tensor:
    """``n_sequences`` validation-partition sequences, split evenly over the sources."""
    from sae_pipeline.data.streaming import configured_sources, pack_token_stream, stream_documents

    sources = configured_sources(cfg.data)
    if n_sequences % len(sources):
        raise ValueError(f"--n-sequences must be a multiple of the {len(sources)} configured sources")
    rows = []
    for source in sources:
        sequences = pack_token_stream(
            stream_documents(cfg.data, source, partition="validation"), tokenizer, seq_len=cfg.data.seq_len
        )
        rows.extend(itertools.islice(sequences, n_sequences // len(sources)))
    if len(rows) != n_sequences:
        raise RuntimeError(f"Only {len(rows)} of {n_sequences} validation sequences are available")
    return torch.stack(rows)


def _hf_extractor(cfg: PipelineCfg, device: str) -> tuple[ActivationExtractor, object]:
    from sae_pipeline.extraction.hf import HFHookExtractor
    from sae_pipeline.model.loader import load_model_and_tokenizer

    model, tokenizer = load_model_and_tokenizer(cfg.model.model_copy(update={"device_map": {"": device}}))
    model.eval()
    requests = site_requests(cfg.target.sites or [])
    return HFHookExtractor(model, requests, model_device=device, capture_device="cpu"), tokenizer


def _vllm_extractor(cfg: PipelineCfg) -> ActivationExtractor:
    from sae_pipeline.extraction.vllm import VLLMExtractor

    specs = {target.slug: ComponentSpec.parse(target.layer, target.component) for target in cfg.target.sites or []}
    return VLLMExtractor.from_config(cfg, specs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--backend", choices=["hf_hooks", "vllm"], default=None)
    parser.add_argument("--token-ids", required=True, help="safetensors file with token_ids [sequences, seq_len]")
    parser.add_argument("--n-sequences", type=int, default=None, help="sample this many if --token-ids is missing")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0", help="model device for the hf_hooks backend")
    parser.add_argument("--batch-sequences", type=int, default=4, help="sequences per extraction call")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = PipelineCfg.from_yaml(args.config)
    if not cfg.target.sites:
        raise SystemExit("dump_activations needs target.sites")
    backend = args.backend or cfg.extraction.backend
    token_path = Path(args.token_ids)

    tokenizer = None
    if backend == "hf_hooks":
        extractor, tokenizer = _hf_extractor(cfg, args.device)
    else:
        extractor = _vllm_extractor(cfg)

    if token_path.exists():
        token_ids = load_file(str(token_path))["token_ids"]
    else:
        if args.n_sequences is None or tokenizer is None:
            raise SystemExit(f"{token_path} does not exist; create it with --backend hf_hooks --n-sequences N")
        token_ids = sample_validation_token_ids(cfg, tokenizer, args.n_sequences)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        save_file({"token_ids": token_ids.contiguous()}, str(token_path))
        log.info("Wrote %d validation sequences to %s", token_ids.shape[0], token_path)

    parts: dict[str, list[torch.Tensor]] = {}
    try:
        for batch in token_ids.split(args.batch_sequences):
            for slug, activations in extractor.extract(batch).activations.items():
                parts.setdefault(slug, []).append(activations.cpu())
    finally:
        extractor.close()

    out = {slug: torch.cat(values).contiguous() for slug, values in parts.items()}
    out["token_ids"] = token_ids.contiguous()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_file(out, args.out, metadata={"backend": backend, "model": cfg.model.name})
    log.info("Wrote %s activations for %d sites to %s", backend, len(parts), args.out)


if __name__ == "__main__":
    run_then_hard_exit(main)
