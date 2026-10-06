"""Activation capture inside vLLM workers, loaded through ``worker_extension_cls``.

vLLM mixes this class into its GPU worker, so the methods run in the worker
process with the model at hand. ``sae_install_hooks`` registers forward hooks
on the requested sites of a NemotronH-style model (``model.layers[L]`` and
``model.layers[L].mixer``). After every forward pass the captured rows, the
step's ``input_ids`` and ``positions`` are written to one safetensors file on
a background thread; ``sae_flush`` waits for the writes and returns the paths.
All names start with ``sae_`` so they cannot collide with worker attributes.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import nn

# Component kind -> substring of the mixer class name it must be hooked on.
MIXER_KINDS = {"mamba_out": "Mamba", "attn_out_prelinear": "Attention", "moe_out": "MoE"}
RESIDUAL_KINDS = ("resid_pre", "resid_post")


def _arg(args: tuple, kwargs: dict, index: int, name: str):
    return kwargs[name] if name in kwargs else args[index]


def _residual(hidden: torch.Tensor, residual: torch.Tensor | None) -> torch.Tensor:
    # vLLM keeps the residual stream split as (mixer output, running residual).
    return hidden if residual is None else hidden + residual


class ActivationHookExtension:
    def _sae_model(self) -> nn.Module:
        # The only vLLM worker internal this extension relies on.
        return self.model_runner.model  # type: ignore[attr-defined]

    def sae_install_hooks(self, sites: list[tuple[str, str, int]], out_dir: str) -> None:
        """Hook every ``(slug, component kind, layer)``; files go to ``out_dir``."""
        model = self._sae_model()
        layers = model.model.layers
        self.sae_out_dir = Path(out_dir)
        self.sae_out_dir.mkdir(parents=True, exist_ok=True)
        self.sae_step = 0
        self.sae_captured: dict[str, torch.Tensor] = {}
        self.sae_inputs: dict[str, torch.Tensor] = {}
        self.sae_writes: list[tuple[Path, Future]] = []
        self.sae_writer = ThreadPoolExecutor(max_workers=1)
        self.sae_handles = [
            model.register_forward_pre_hook(self._sae_record_inputs, with_kwargs=True),
            model.register_forward_hook(self._sae_write_step),
        ]
        for slug, kind, layer in sites:
            if kind == "resid_pre":
                handle = layers[layer].register_forward_pre_hook(partial(self._sae_capture_resid_pre, slug), with_kwargs=True)
            elif kind == "resid_post":
                handle = layers[layer].register_forward_hook(partial(self._sae_capture_resid_post, slug))
            elif kind in MIXER_KINDS:
                mixer = layers[layer].mixer
                if MIXER_KINDS[kind] not in type(mixer).__name__:
                    raise ValueError(f"{slug}: layer {layer} has a {type(mixer).__name__}, not a {MIXER_KINDS[kind]} block")
                handle = mixer.register_forward_hook(partial(self._sae_capture_output, slug))
            else:
                raise ValueError(f"{slug}: component {kind!r} has no hook point in vLLM")
            self.sae_handles.append(handle)

    def sae_flush(self) -> list[str]:
        """Wait for pending writes; return the files written since the last flush."""
        paths = []
        for path, write in self.sae_writes:
            write.result()
            paths.append(str(path))
        self.sae_writes = []
        return paths

    def sae_remove_hooks(self) -> None:
        self.sae_flush()
        for handle in self.sae_handles:
            handle.remove()
        self.sae_writer.shutdown()

    def _sae_record_inputs(self, _module, args, kwargs) -> None:
        self.sae_inputs = {
            "input_ids": _arg(args, kwargs, 0, "input_ids"),
            "positions": _arg(args, kwargs, 1, "positions"),
        }

    def _sae_capture_resid_pre(self, slug, _module, args, kwargs) -> None:
        self.sae_captured[slug] = _residual(_arg(args, kwargs, 1, "hidden_states"), _arg(args, kwargs, 2, "residual"))

    def _sae_capture_resid_post(self, slug, _module, _args, output) -> None:
        self.sae_captured[slug] = _residual(*output)

    def _sae_capture_output(self, slug, _module, _args, output) -> None:
        self.sae_captured[slug] = output

    def _sae_write_step(self, _module, _args, _output) -> None:
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return
        tensors = {**self.sae_inputs, **self.sae_captured}
        cpu = {name: tensor.detach().to("cpu").contiguous() for name, tensor in tensors.items()}
        self.sae_captured = {}
        path = self.sae_out_dir / f"step_{self.sae_step:08d}.safetensors"
        self.sae_step += 1
        self.sae_writes.append((path, self.sae_writer.submit(save_file, cpu, str(path))))
