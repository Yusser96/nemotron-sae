"""Evaluate SAEs on two activation dumps of the same tokens and compare the metrics.

Complements ``cli.compare_activations``: instead of asking whether two
backends' activations agree, it asks whether the difference matters to an
SAE -- FVU, L0, dead latents and dictionary use on each dump, and the deltas.
SAE weights must be in raw activation units (e.g. released raw-export
checkpoints), since the dumps hold raw activations.

Usage:
    python -m sae_pipeline.cli.compare_sae_metrics hf.safetensors vllm.safetensors \\
        --sae L12_resid_post=path/to/sae_weights.safetensors --out sae_report.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from sae_pipeline.eval.metrics import ReconMetrics, streaming_reconstruction_metrics
from sae_pipeline.sae.checkpoint import load_sae_weights
from sae_pipeline.sae.train import build_sae

_REPORTED = ("fvu", "l0", "dead_pct", "d_is", "u_is")


def load_raw_sae(path: str | Path, *, device: str = "cpu"):
    """A JumpReLU SAE whose weights apply directly to raw activations."""
    with safe_open(str(path), framework="pt") as handle:
        d_in, d_sae = handle.get_slice("W_enc").get_shape()
    sae = build_sae("jumprelu", d_in=d_in, d_sae=d_sae, pre_encoder_bias=False)
    load_sae_weights(sae, path)
    return sae.to(device).eval()


def sae_metrics(sae, activations: torch.Tensor, *, batch_size: int, device: str) -> ReconMetrics:
    batches = (chunk.to(device, dtype=torch.float32) for chunk in activations.split(batch_size))
    metrics = streaming_reconstruction_metrics(sae, batches, dead_threshold_tokens=activations.shape[0])
    assert isinstance(metrics, ReconMetrics)
    return metrics


def compare_sae_metrics(
    reference_path: str | Path,
    candidate_path: str | Path,
    saes: dict[str, str | Path],
    *,
    batch_size: int = 4096,
    device: str = "cpu",
) -> dict[str, dict[str, object]]:
    reference = load_file(str(reference_path))
    candidate = load_file(str(candidate_path))
    if not torch.equal(reference["token_ids"], candidate["token_ids"]):
        raise ValueError("The dumps were made from different token ids")
    report = {}
    for slug, weights in saes.items():
        if slug not in reference or slug not in candidate:
            raise KeyError(f"{slug} is not in both dumps")
        sae = load_raw_sae(weights, device=device)
        with torch.no_grad():
            ref = sae_metrics(sae, reference[slug], batch_size=batch_size, device=device)
            cand = sae_metrics(sae, candidate[slug], batch_size=batch_size, device=device)
        ref_values = {key: asdict(ref)[key] for key in _REPORTED}
        cand_values = {key: asdict(cand)[key] for key in _REPORTED}
        report[slug] = {
            "sae": str(weights),
            "reference": ref_values,
            "candidate": cand_values,
            "delta": {key: cand_values[key] - ref_values[key] for key in _REPORTED},
        }
    return report


def _parse_sae(value: str) -> tuple[str, str]:
    slug, sep, path = value.partition("=")
    if not sep or not slug or not path:
        raise argparse.ArgumentTypeError("expected SLUG=PATH")
    return slug, path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--sae", action="append", type=_parse_sae, required=True, help="SLUG=raw-unit SAE weights")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    report = compare_sae_metrics(
        args.reference, args.candidate, dict(args.sae), batch_size=args.batch_size, device=args.device
    )
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
