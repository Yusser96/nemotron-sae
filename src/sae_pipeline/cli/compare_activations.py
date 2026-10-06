"""Compare two activation dumps (see ``cli.dump_activations``) site by site.

Exits non-zero unless every shared site passes the agreement gate.

Usage:
    python -m sae_pipeline.cli.compare_activations hf.safetensors vllm.safetensors --out report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from sae_pipeline.extraction.compare import compare_activations


def compare_dumps(
    reference_path: str | Path,
    candidate_path: str | Path,
    *,
    min_cosine_mean: float = 0.999,
    min_cosine_p01: float = 0.99,
    max_relative_l2: float = 2e-2,
) -> dict[str, object]:
    reference = load_file(str(reference_path))
    candidate = load_file(str(candidate_path))
    if not torch.equal(reference["token_ids"], candidate["token_ids"]):
        raise ValueError("The dumps were made from different token ids")
    gate = dict(min_cosine_mean=min_cosine_mean, min_cosine_p01=min_cosine_p01, max_relative_l2=max_relative_l2)
    sites = sorted((reference.keys() & candidate.keys()) - {"token_ids"})
    if not sites:
        raise ValueError("The dumps share no sites")
    per_site = {}
    for slug in sites:
        agreement = compare_activations(reference[slug], candidate[slug])
        per_site[slug] = {**agreement.as_dict(), "passes": agreement.passes(**gate)}
    metadata = {}
    for name, path in (("reference", reference_path), ("candidate", candidate_path)):
        with safe_open(str(path), framework="pt") as handle:
            metadata[name] = handle.metadata()
    return {
        "gate": gate,
        "metadata": metadata,
        "n_sequences": int(reference["token_ids"].shape[0]),
        "only_in_reference": sorted(reference.keys() - candidate.keys()),
        "only_in_candidate": sorted(candidate.keys() - reference.keys()),
        "sites": per_site,
        "passes": all(site["passes"] for site in per_site.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--out", default=None, help="write the JSON report here as well")
    parser.add_argument("--min-cosine-mean", type=float, default=0.999)
    parser.add_argument("--min-cosine-p01", type=float, default=0.99)
    parser.add_argument("--max-relative-l2", type=float, default=2e-2)
    args = parser.parse_args()

    report = compare_dumps(
        args.reference,
        args.candidate,
        min_cosine_mean=args.min_cosine_mean,
        min_cosine_p01=args.min_cosine_p01,
        max_relative_l2=args.max_relative_l2,
    )
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
    sys.exit(0 if report["passes"] else 1)


if __name__ == "__main__":
    main()
