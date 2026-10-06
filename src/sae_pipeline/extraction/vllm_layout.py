"""Map residual-stream sites onto vLLM auxiliary hidden-state layer ids.

vLLM's hidden-state extraction captures the residual stream at decoder-layer
boundaries: id 0 is the input to layer 0 (the embeddings) and id ``L + 1`` is
the output of layer ``L``. So ``resid_pre(L)`` is id ``L``, ``resid_post(L)``
is id ``L + 1``, and ``resid_post(L)`` / ``resid_pre(L + 1)`` share one id.
The ids are relative to the first local layer, so this holds only without
pipeline parallelism. Components inside a layer (Mamba, attention, MoE
outputs, single experts) are not at a boundary and are not supported.
"""

from __future__ import annotations

from dataclasses import dataclass

from sae_pipeline.extraction.base import UnsupportedComponentError
from sae_pipeline.hooks.components import ComponentSpec

_BOUNDARY_OFFSET = {"resid_pre": 0, "resid_post": 1}


def residual_layer_id(spec: ComponentSpec, num_hidden_layers: int) -> int:
    """The vLLM aux hidden-state id holding ``spec``'s activations."""
    if spec.kind not in _BOUNDARY_OFFSET or spec.expert_idx is not None or spec.head_idx is not None:
        raise UnsupportedComponentError(
            f"vLLM hidden-state extraction only covers residual-stream sites "
            f"(resid_pre/resid_post), not {spec.slug}"
        )
    if not 0 <= spec.layer < num_hidden_layers:
        raise ValueError(f"{spec.slug}: layer {spec.layer} is outside the model's {num_hidden_layers} layers")
    return spec.layer + _BOUNDARY_OFFSET[spec.kind]


@dataclass(frozen=True)
class VLLMLayout:
    """Which aux layers to request from vLLM, and where each site's column is."""

    layer_ids: tuple[int, ...]  # sorted, unique: the order of hidden_states[:, i, :]
    columns: dict[str, int]  # slug -> index into layer_ids

    @classmethod
    def from_specs(cls, specs: dict[str, ComponentSpec], num_hidden_layers: int) -> "VLLMLayout":
        ids = {slug: residual_layer_id(spec, num_hidden_layers) for slug, spec in specs.items()}
        layer_ids = tuple(sorted(set(ids.values())))
        position = {layer_id: index for index, layer_id in enumerate(layer_ids)}
        return cls(layer_ids=layer_ids, columns={slug: position[layer_id] for slug, layer_id in ids.items()})
