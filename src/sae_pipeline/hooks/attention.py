"""Locate the attention output projection without guessing activation semantics."""

from torch import nn


def output_projection(module: nn.Module) -> nn.Module:
    for name in ("o_proj", "out_proj", "dense"):
        projection = getattr(module, name, None)
        if isinstance(projection, nn.Module):
            return projection
    if isinstance(module, nn.Linear):
        return module
    raise ValueError(f"No attention output projection in {type(module).__name__}; specify its exact hook path")


def head_count(module: nn.Module) -> int | None:
    for owner in (module, getattr(module, "config", None)):
        for name in ("num_heads", "num_attention_heads", "n_heads"):
            count = getattr(owner, name, None)
            if isinstance(count, int) and count > 0:
                return count
    return None
