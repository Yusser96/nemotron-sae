"""Reference extraction backend: PyTorch forward hooks on a transformers model."""

from __future__ import annotations

import torch
from torch import nn

from sae_pipeline.extraction.base import ActivationBatch
from sae_pipeline.hooks.components import ComponentSpec
from sae_pipeline.hooks.extractor import capture_many
from sae_pipeline.model.topology import HookSite


class HFHookExtractor:
    """Capture every requested site from one eager forward pass (see ``hooks.extractor``)."""

    def __init__(
        self,
        model: nn.Module,
        requests: dict[str, tuple[ComponentSpec, HookSite]],
        *,
        model_device: str | torch.device,
        capture_device: str | torch.device = "cpu",
    ) -> None:
        self.model = model
        self.requests = requests
        self.model_device = model_device
        self.capture_device = capture_device

    def supports(self, spec: ComponentSpec) -> bool:
        return True

    def extract(self, token_ids: torch.Tensor) -> ActivationBatch:
        with capture_many(self.model, self.requests, device=self.capture_device) as buffers:
            with torch.no_grad():
                self.model(token_ids.to(self.model_device))
            activations = {slug: buffer.consume() for slug, buffer in buffers.items()}
        return ActivationBatch(token_ids=token_ids, activations=activations)

    def close(self) -> None:
        pass
