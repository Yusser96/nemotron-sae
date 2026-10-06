"""Collect the activation files written by the vLLM worker hooks (see ``vllm_worker``).

The sink directory can be a RAM-backed tmpfs (streaming) or a disk; with the
``consume`` lifecycle every file is deleted once its rows have been read.
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

    def collect(
        self,
        step_paths: list[str],
        token_ids: torch.Tensor,
        slugs: list[str],
        *,
        device: str | torch.device,
    ) -> dict[str, torch.Tensor]:
        """Per-site ``(sequences * seq_len, d)`` activations for ``token_ids``, in row order.

        Each step file holds the rows of one forward pass: whole prompts back
        to back, each starting at position 0. Rows are matched to the
        submitted prompts by their token ids; rows that match no prompt (e.g.
        padding) are ignored.
        """
        pending: dict[tuple[int, ...], list[int]] = {}
        for index, row in enumerate(token_ids.detach().cpu().tolist()):
            pending.setdefault(tuple(row), []).append(index)
        found: dict[int, dict[str, torch.Tensor]] = {}
        for step_path in step_paths:
            step = load_file(step_path)
            ids, positions = step["input_ids"], step["positions"]
            starts = (positions == 0).nonzero().flatten().tolist() + [positions.numel()]
            for start, end in zip(starts, starts[1:]):
                matches = pending.get(tuple(ids[start:end].tolist()))
                if matches:
                    found[matches.pop(0)] = {slug: step[slug][start:end] for slug in slugs}
            if self.lifecycle == "consume":
                Path(step_path).unlink()
        missing = sorted(set(range(token_ids.shape[0])) - found.keys())
        if missing:
            raise RuntimeError(f"No captured activations for submitted sequences {missing}")
        return {
            slug: torch.cat([found[index][slug] for index in range(token_ids.shape[0])]).to(device)
            for slug in slugs
        }

    def cleanup(self) -> None:
        """Remove files a consuming run left behind (e.g. after an interruption)."""
        if self.lifecycle != "consume" or not self.path.exists():
            return
        for leftover in self.path.glob("*.safetensors"):
            leftover.unlink()
        if not any(self.path.iterdir()):
            self.path.rmdir()
