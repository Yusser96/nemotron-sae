"""Activation capture using PyTorch forward hooks.

We deliberately use raw forward hooks (not nnsight tracing) because:
- The model uses trust_remote_code; nnsight wraps it fine, but plain hooks are simpler
  and have zero overhead in eager mode.
- For per-expert hooks we need to slice on the routed-token indices, which is awkward
  to express in nnsight's tracing API.

The extractor returns a tensor of shape (n_tokens_seen, d_activation).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
import torch.nn as nn

from sae_pipeline.hooks.components import ComponentSpec
from sae_pipeline.hooks.attention import output_projection, head_count
from sae_pipeline.model.topology import HookSite


def _resolve_module(model: nn.Module, dotted: str) -> nn.Module:
    obj: nn.Module | object = model
    for part in dotted.split("."):
        if part.isdigit():
            obj = obj[int(part)]   # type: ignore[index]
        else:
            obj = getattr(obj, part)
    return obj  # type: ignore[return-value]


def _resolve_block(model: nn.Module, layer: int) -> nn.Module:
    """Return the layer's outer block. Tries .model.layers[L], .layers[L], etc."""
    for path in (
        f"model.layers.{layer}",
        f"layers.{layer}",
        f"transformer.h.{layer}",
        f"backbone.layers.{layer}",
    ):
        try:
            return _resolve_module(model, path)
        except (AttributeError, IndexError):
            continue
    raise KeyError(f"Could not locate transformer block for layer {layer}.")


class _BufferList:
    """Accumulates captured activations into a list of tensors on one device."""

    def __init__(self, device: str | torch.device = "cpu") -> None:
        self.device = device
        self.buf: list[torch.Tensor] = []

    def add(self, x: torch.Tensor) -> None:
        # Flatten (B, T, D) -> (B*T, D) and move to the target device.
        if x.ndim >= 2:
            flat = x.reshape(-1, x.shape[-1])
        else:
            flat = x
        self.buf.append(flat.detach().to(self.device, copy=True))

    def consume(self) -> torch.Tensor:
        """Return accumulated tensor and reset the buffer."""
        if not self.buf:
            return torch.empty(0)
        out = torch.cat(self.buf, dim=0)
        self.buf.clear()
        return out


def _capture_handle(
    model: nn.Module,
    spec: ComponentSpec,
    site: HookSite,
    buffer: _BufferList,
) -> torch.utils.hooks.RemovableHandle:
    """Register one capture hook and return its removable handle."""
    if spec.kind in {"resid_pre", "resid_post"}:
        block = _resolve_block(model, spec.layer)
        if spec.kind == "resid_pre":
            def pre_hook(_mod, args, _kwargs, buffer=buffer):
                # First positional arg is hidden_states for most HF blocks.
                hidden = args[0] if args else _kwargs.get("hidden_states")
                if hidden is None:
                    raise ValueError("resid_pre requires hidden_states as a positional or keyword argument")
                buffer.add(hidden)
                return None
            return block.register_forward_pre_hook(pre_hook, with_kwargs=True)
        else:
            def post_hook(_mod, _args, output, buffer=buffer):
                # Output is usually a tensor or (tensor, ...) tuple.
                hs = output[0] if isinstance(output, tuple) else output
                buffer.add(hs)
            return block.register_forward_hook(post_hook)
    else:
        mod = _resolve_module(model, site.module_path)
        if spec.kind in {"attn_out_prelinear", "attn_head"}:
            projection = output_projection(mod)
            heads = (site.extra or {}).get("num_heads") or head_count(mod)
            if spec.kind == "attn_head" and (not heads or spec.head_idx is None or spec.head_idx >= int(heads)):
                raise ValueError(f"Cannot resolve attention head {spec.head_idx}; supply verified num_heads metadata")
            def projection_input(_mod, args, kwargs, buffer=buffer):
                hidden = args[0] if args else kwargs.get("hidden_states", kwargs.get("input"))
                if hidden is None:
                    raise ValueError("Attention output projection received no input tensor")
                if spec.kind == "attn_head":
                    if hidden.shape[-1] % int(heads):
                        raise ValueError("Attention projection width is not divisible by num_heads")
                    width = hidden.shape[-1] // int(heads)
                    hidden = hidden[..., spec.head_idx * width : (spec.head_idx + 1) * width]
                buffer.add(hidden)
            return projection.register_forward_pre_hook(projection_input, with_kwargs=True)
        def fwd_hook(_mod, _args, output, buffer=buffer):
            t = output[0] if isinstance(output, tuple) else output
            buffer.add(t)
        return mod.register_forward_hook(fwd_hook)


@contextmanager
def capture(
    model: nn.Module,
    spec: ComponentSpec,
    site: HookSite,
    device: str | torch.device = "cpu",
) -> Iterator[_BufferList]:
    """Register a forward hook for `spec` (resolved to `site`) and yield a buffer.

    Caller runs the model forward inside the `with`, then calls buffer.consume()
    to retrieve the (n_tokens, d) tensor.
    """
    buf = _BufferList(device=device)
    handle = _capture_handle(model, spec, site, buf)

    try:
        yield buf
    finally:
        handle.remove()


@contextmanager
def capture_many(
    model: nn.Module,
    requests: dict[str, tuple[ComponentSpec, HookSite]],
    device: str | torch.device = "cpu",
) -> Iterator[dict[str, _BufferList]]:
    """Capture several sites during one frozen-model forward pass.

    ``device`` lets a consumer that trains on the GPU capture straight onto
    it instead of paying a CPU round trip.
    """
    buffers = {slug: _BufferList(device=device) for slug in requests}
    handles: list[torch.utils.hooks.RemovableHandle] = []
    try:
        for slug, (spec, site) in requests.items():
            handles.append(_capture_handle(model, spec, site, buffers[slug]))
        yield buffers
    finally:
        for handle in handles:
            handle.remove()
