"""Read the activation files written by the vLLM worker hooks (see ``vllm_worker``).

The sink directory can be a RAM-backed tmpfs (streaming) or a disk; with the
``consume`` lifecycle every file is deleted once it has been read.
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file

from sae_pipeline.config import ActivationSinkCfg


class ActivationSink:
    def __init__(self, cfg: ActivationSinkCfg) -> None:
        self.path = cfg.resolved_path()
        self.lifecycle = cfg.lifecycle

    def prepare(self) -> Path:
        self.path.mkdir(parents=True, exist_ok=True)
        return self.path

    def read(self, step_paths: list[str]) -> list[dict[str, torch.Tensor]]:
        """Load step files in order; ``consume`` deletes each once loaded."""
        steps = []
        for step_path in step_paths:
            steps.append(load_file(step_path))
            if self.lifecycle == "consume":
                Path(step_path).unlink()
        return steps

    def cleanup(self) -> None:
        """Remove files a consuming run left behind (e.g. after an interruption)."""
        if self.lifecycle != "consume" or not self.path.exists():
            return
        for leftover in self.path.glob("*.safetensors"):
            leftover.unlink()
        if not any(self.path.iterdir()):
            self.path.rmdir()
