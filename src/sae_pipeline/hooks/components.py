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


def site_requests(sites) -> dict[str, tuple[ComponentSpec, HookSite]]:
    """Build each site's (ComponentSpec, HookSite) request from its config entry.

    Each entry already names its exact hook (component, layer, hook_name), so
    this builds the HookSite directly rather than resolving it from enumerated
    topology: topology resolution can guess the wrong module for a custom
    component path (e.g. ``.mixer``), while released SAE metadata is
    authoritative. ``sites`` duck-types ``target.slug/.layer/.component/.hook_name``
    (a ``list[HookTargetCfg]``, typically ``cfg.target.sites``).
    """
    return {
        target.slug: (
            ComponentSpec.parse(target.layer, target.component),
            HookSite(target.component, target.layer, target.hook_name),
        )
        for target in sites
    }


def select_sites(sites, slugs: list[str] | None) -> list:
    """The entries of ``sites`` named in ``slugs`` (all of them for None), in config order."""
    if slugs is None:
        return list(sites)
    unknown = sorted(set(slugs) - {target.slug for target in sites})
    if unknown:
        raise ValueError(f"Sites {unknown} are not in target.sites")
    return [target for target in sites if target.slug in slugs]


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
