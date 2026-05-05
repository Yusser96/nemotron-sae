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
    """Accumulates captured activations into a list of CPU tensors."""

    def __init__(self) -> None:
        self.buf: list[torch.Tensor] = []

    def add(self, x: torch.Tensor) -> None:
        # Flatten (B, T, D) -> (B*T, D) and move to CPU pinned memory.
        if x.ndim >= 2:
            flat = x.reshape(-1, x.shape[-1])
        else:
            flat = x
        self.buf.append(flat.detach().to("cpu", copy=True))

    def consume(self) -> torch.Tensor:
        """Return accumulated tensor and reset the buffer."""
        if not self.buf:
            return torch.empty(0)
        out = torch.cat(self.buf, dim=0)
        self.buf.clear()
        return out


@contextmanager
def capture(model: nn.Module, spec: ComponentSpec, site: HookSite) -> Iterator[_BufferList]:
    """Register a forward hook for `spec` (resolved to `site`) and yield a buffer.

    Caller runs the model forward inside the `with`, then calls buffer.consume()
    to retrieve the (n_tokens, d) tensor.
    """
    buf = _BufferList()
    handles: list[torch.utils.hooks.RemovableHandle] = []

    if spec.kind in {"resid_pre", "resid_post"}:
        block = _resolve_block(model, spec.layer)
        if spec.kind == "resid_pre":
            def pre_hook(_mod, args, _kwargs):
                # First positional arg is hidden_states for most HF blocks.
                if args:
                    buf.add(args[0])
                return None
            handles.append(block.register_forward_pre_hook(pre_hook, with_kwargs=True))
        else:
            def post_hook(_mod, _args, output):
                # Output is usually a tensor or (tensor, ...) tuple.
                hs = output[0] if isinstance(output, tuple) else output
                buf.add(hs)
            handles.append(block.register_forward_hook(post_hook))
    elif spec.kind == "expert":
        mod = _resolve_module(model, site.module_path)
        def fwd_hook(_mod, _args, output):
            t = output[0] if isinstance(output, tuple) else output
            buf.add(t)
        handles.append(mod.register_forward_hook(fwd_hook))
    else:
        # Generic case: hook the resolved module's forward output.
        mod = _resolve_module(model, site.module_path)
        def fwd_hook(_mod, _args, output):
            t = output[0] if isinstance(output, tuple) else output
            buf.add(t)
        handles.append(mod.register_forward_hook(fwd_hook))

    try:
        yield buf
    finally:
        for h in handles:
            h.remove()
