"""Resolve a (layer, component) spec to a concrete HookSite using a topology JSON.

This decouples user-facing component names ("resid_post", "moe_out", "expert.42")
from the model-internal module paths, which depend on trust_remote_code internals.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from sae_pipeline.model.topology import HookSite


@dataclass
class ComponentSpec:
    layer: int
    kind: str            # resid_pre|resid_post|mamba_out|attn_out_prelinear|moe_out|expert|shared_expert|attn_head
    expert_idx: int | None = None
    head_idx: int | None = None

    @classmethod
    def parse(cls, layer: int, component: str) -> "ComponentSpec":
        """Parse "expert.42" / "attn_head.7" / "resid_post" / etc."""
        if "." in component:
            kind, idx_s = component.split(".", 1)
            idx = int(idx_s)
            if kind == "expert":
                return cls(layer=layer, kind="expert", expert_idx=idx)
            if kind == "attn_head":
                return cls(layer=layer, kind="attn_head", head_idx=idx)
            raise ValueError(f"Unknown indexed component {component!r}")
        return cls(layer=layer, kind=component)

    @property
    def slug(self) -> str:
        """Filesystem-safe identifier (no zero-padding so it matches `L${LAYER}` shell vars)."""
        if self.expert_idx is not None:
            return f"L{self.layer}_expert_{self.expert_idx}"
        if self.head_idx is not None:
            return f"L{self.layer}_head_{self.head_idx}"
        return f"L{self.layer}_{self.kind}"


def load_topology(path: str | Path) -> list[HookSite]:
    with open(path, "r") as f:
        payload = json.load(f)
    return [HookSite(**s) for s in payload["sites"]]


def resolve(spec: ComponentSpec, sites: list[HookSite]) -> HookSite:
    """Find the matching HookSite. Raises if 0 or >1 match."""
    matches: list[HookSite] = []
    for s in sites:
        if s.layer != spec.layer:
            continue
        if s.component_kind != spec.kind:
            continue
        if spec.expert_idx is not None:
            if not s.extra or s.extra.get("expert_idx") != spec.expert_idx:
                continue
        matches.append(s)

    if not matches:
        raise KeyError(f"No HookSite for {spec}.")
    if len(matches) > 1:
        # Disambiguate: prefer the shortest module path (most "outer" candidate).
        matches.sort(key=lambda s: len(s.module_path))
    return matches[0]
