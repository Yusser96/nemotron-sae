"""Walk a loaded HF model and enumerate hookable activation sites.

For Nemotron-3-Nano (hybrid Mamba-2 + Attention + MoE) the exact module paths come
from the trust_remote_code modeling file, not docs. We classify modules by name pattern
and dump a `model_topology.json` so subsequent jobs reference verified names.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import torch.nn as nn

log = logging.getLogger(__name__)


@dataclass
class HookSite:
    """A single hookable point in the model."""
    component_kind: str          # resid_pre | resid_post | mamba_out | attn_out_prelinear |
                                 # attn_head | moe_out | expert | shared_expert
    layer: int
    module_path: str             # dotted path from the model root
    extra: dict[str, int | str] | None = None   # e.g. {"expert_idx": 42}


# Compiled patterns for classifying module names. The model is a custom Nemotron arch,
# so we cast a wide net and keep only what looks load-bearing.
_LAYER_RE = re.compile(r"\.layers?\.(\d+)\b")
_EXPERT_RE = re.compile(r"\.(?:experts?|moe_experts?)\.(\d+)\b")
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("expert", re.compile(r"\.(?:experts?|moe_experts?)\.\d+$")),
    ("shared_expert", re.compile(r"\.(?:shared_experts?|shared_mlp)(\.\d+)?$")),
    ("moe_out", re.compile(r"\.(?:block_sparse_moe|moe|mlp_moe|sparse_moe)$")),
    ("mamba_out", re.compile(r"\.(?:mamba|mamba2|mixer|ssm)$")),
    ("attn_out_prelinear", re.compile(
        r"\.(?:self_attn|attention|attn)$"
    )),
]


def _classify(name: str) -> str | None:
    for kind, rx in _PATTERNS:
        if rx.search(name):
            return kind
    return None


def _layer_of(name: str) -> int | None:
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def _expert_idx(name: str) -> int | None:
    m = _EXPERT_RE.search(name)
    return int(m.group(1)) if m else None


def enumerate_hooks(model: nn.Module) -> list[HookSite]:
    """Walk every named submodule and return a list of HookSite entries."""
    sites: list[HookSite] = []
    n_layers = 0
    for name, _ in model.named_modules():
        layer = _layer_of(name)
        if layer is None:
            continue
        n_layers = max(n_layers, layer + 1)
        kind = _classify(name)
        if kind is None:
            continue
        extra: dict[str, int | str] | None = None
        if kind == "expert":
            ei = _expert_idx(name)
            if ei is None:
                continue
            extra = {"expert_idx": ei}
        sites.append(HookSite(component_kind=kind, layer=layer, module_path=name, extra=extra))

    # Residual stream sites are *block I/O*, not modules; we synthesize one per layer.
    for L in range(n_layers):
        sites.append(HookSite(component_kind="resid_pre", layer=L, module_path=f"<residual:{L}:pre>"))
        sites.append(HookSite(component_kind="resid_post", layer=L, module_path=f"<residual:{L}:post>"))

    sites.sort(key=lambda s: (s.layer, s.component_kind, s.module_path))
    return sites


def dump_topology(sites: list[HookSite], out_path: str | Path) -> dict:
    """Write a JSON manifest of all hookable sites and return a summary dict."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "n_sites": len(sites), "sites": [asdict(s) for s in sites]}
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    by_kind: dict[str, int] = {}
    for s in sites:
        by_kind[s.component_kind] = by_kind.get(s.component_kind, 0) + 1
    summary = {"n_sites": len(sites), "by_kind": by_kind}
    log.info("Wrote topology to %s: %s", out_path, summary)
    return summary


def cli() -> None:
    """`python -m sae_pipeline.model.topology --model <name>` prints/dumps the topology."""
    import argparse

    parser = argparse.ArgumentParser(description="Enumerate hookable sites in an HF causal LM.")
    parser.add_argument("--model", required=True, help="HF model id")
    parser.add_argument("--out", default="outputs/model_topology.json")
    parser.add_argument("--print", action="store_true", help="Print summary to stdout")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--no-trust-remote-code", action="store_true")
    parser.add_argument("--device-map", default="auto")
    args = parser.parse_args()

    from sae_pipeline.config import ModelCfg
    from sae_pipeline.model.loader import load_model_and_tokenizer

    cfg = ModelCfg(
        name=args.model,
        dtype=args.dtype,
        trust_remote_code=not args.no_trust_remote_code,
        device_map=args.device_map,
    )
    model, _ = load_model_and_tokenizer(cfg)
    sites = enumerate_hooks(model)
    summary = dump_topology(sites, args.out)
    if args.print:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    cli()
