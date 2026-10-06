"""Read the hidden-state files written by vLLM's hidden-states connector.

Each request produces one safetensors file holding ``token_ids [T']`` and
``hidden_states [T', n_layers, d]``, where ``T'`` may also cover generated
tokens. The sink directory can be a RAM-backed tmpfs (streaming) or a disk
(offline extraction); ``lifecycle`` decides whether files outlive a read.
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file

from sae_pipeline.config import ActivationSinkCfg
from sae_pipeline.extraction.vllm_layout import VLLMLayout


class HiddenStateSink:
    def __init__(self, cfg: ActivationSinkCfg) -> None:
        self.path = cfg.resolved_path()
        self.lifecycle = cfg.lifecycle

    def prepare(self) -> Path:
        self.path.mkdir(parents=True, exist_ok=True)
        return self.path

    def read(
        self,
        path: str | Path,
        *,
        expected_token_ids: torch.Tensor,
        layout: VLLMLayout,
        device: str | torch.device,
    ) -> dict[str, torch.Tensor]:
        """Per-site ``(T, d)`` activations for the ``T`` submitted tokens of one request."""
        path = Path(path)
        tensors = load_file(str(path))
        token_ids, hidden = tensors["token_ids"], tensors["hidden_states"]
        n_tokens = expected_token_ids.numel()
        if hidden.ndim != 3 or hidden.shape[1] != len(layout.layer_ids):
            raise ValueError(
                f"{path}: hidden_states shape {tuple(hidden.shape)} does not hold "
                f"{len(layout.layer_ids)} layers"
            )
        if token_ids.shape[0] < n_tokens or not torch.equal(
            token_ids[:n_tokens].to(torch.long), expected_token_ids.detach().cpu().to(torch.long)
        ):
            raise ValueError(f"{path}: token_ids do not match the submitted prompt")
        activations = {
            slug: hidden[:n_tokens, column].to(device, copy=True) for slug, column in layout.columns.items()
        }
        if self.lifecycle == "consume":
            path.unlink()
        return activations

    def cleanup(self) -> None:
        """Remove files a consuming run left behind (e.g. after an interruption)."""
        if self.lifecycle != "consume" or not self.path.exists():
            return
        for leftover in self.path.glob("*.safetensors"):
            leftover.unlink()
        if not any(self.path.iterdir()):
            self.path.rmdir()
