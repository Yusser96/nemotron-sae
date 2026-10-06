"""Read the hidden-state files written by vLLM's hidden-states connector.

Each request produces one safetensors file holding ``token_ids [T']`` and
``hidden_states [T', n_layers, d]``, where ``T'`` may also cover generated
tokens. The sink directory can be a RAM-backed tmpfs (streaming) or a disk
(offline extraction); ``lifecycle`` decides whether files outlive a read.

The connector writes files on background threads, so a path can be returned
before its file exists. With ``synchronization_lock`` the writer holds an
exclusive ``flock`` on ``<file>.lock`` until the file is complete; readers
take a shared lock on it first.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from pathlib import Path

import torch
from safetensors.torch import load_file

from sae_pipeline.config import ActivationSinkCfg
from sae_pipeline.extraction.vllm_layout import VLLMLayout

try:
    import fcntl
except ImportError:  # Windows: vLLM does not run there, so no writer to wait for.
    fcntl = None  # type: ignore[assignment]


def _lock_path(path: Path) -> Path:
    return path.with_name(path.name + ".lock")


@contextlib.contextmanager
def _written(path: Path) -> Iterator[None]:
    """Hold a shared lock on ``path``'s lock file, i.e. wait until the writer is done."""
    lock_path = _lock_path(path)
    if fcntl is None or not lock_path.exists():
        yield
        return
    with open(lock_path) as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


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
        with _written(path):
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
            _lock_path(path).unlink(missing_ok=True)
        return activations

    def collect(
        self,
        step_paths: list[str],
        token_ids: torch.Tensor,
        slugs: list[str],
        *,
        device: str | torch.device,
    ) -> dict[str, torch.Tensor]:
        """Per-site ``(sequences * seq_len, d)`` activations for ``token_ids``, in row order.

        Each step file (see ``vllm_worker``) holds the rows of one forward pass:
        whole prompts back to back, each starting at position 0. Rows are matched
        to the submitted prompts by their token ids; rows that match no prompt
        (e.g. padding) are ignored.
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
        for pattern in ("*.safetensors", "*.safetensors.lock"):
            for leftover in self.path.glob(pattern):
                leftover.unlink()
        if not any(self.path.iterdir()):
            self.path.rmdir()
