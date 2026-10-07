"""Activation capture inside vLLM workers, loaded through ``worker_extension_cls``.

vLLM mixes this class into its GPU worker, so the methods run in the worker
process with the model at hand. ``sae_install_hooks`` registers forward hooks
on the requested sites of a NemotronH-style model (``model.layers[L]`` and
``model.layers[L].mixer``). After every forward pass the captured rows, the
step's ``input_ids`` and ``positions`` either are written to one safetensors
file on a background thread (``sae_flush`` waits for the writes and returns
the paths) or stay on the GPU until ``sae_send`` hands them to a trainer
process (see ``remote``) or ``sae_save`` writes them as one fill per site. All names start with ``sae_`` so they cannot collide with worker attributes.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import nn

from sae_pipeline.extraction.remote import PRODUCER_RANK, SITES_KEY, TRAINER_RANK, open_store, peer_group
from sae_pipeline.hooks.components import ComponentSpec

# Component kind -> substring of the mixer class name it must be hooked on.
MIXER_KINDS = {"mamba_out": "Mamba", "attn_out_prelinear": "Attention", "moe_out": "MoE"}
RESIDUAL_KINDS = ("resid_pre", "resid_post")


def hookable(spec: ComponentSpec) -> bool:
    """Whether a site has a module boundary to hook in vLLM's NemotronH."""
    return spec.expert_idx is None and spec.head_idx is None and (spec.kind in RESIDUAL_KINDS or spec.kind in MIXER_KINDS)


def match_rows(
    steps: Iterable[dict[str, torch.Tensor]], token_ids: torch.Tensor
) -> list[tuple[dict[str, torch.Tensor], int, int]]:
    """For each prompt of ``token_ids``, in order: the step holding its rows and their span.

    Each step holds the rows of one forward pass: whole prompts back to back,
    each starting at position 0. Rows are matched to the submitted prompts by
    their token ids; rows that match no prompt (e.g. padding) are ignored.
    """
    pending: dict[tuple[int, ...], list[int]] = {}
    for index, row in enumerate(token_ids.detach().cpu().tolist()):
        pending.setdefault(tuple(row), []).append(index)
    found: dict[int, tuple[dict[str, torch.Tensor], int, int]] = {}
    for step in steps:
        ids, positions = step["input_ids"].cpu(), step["positions"].cpu()
        starts = (positions == 0).nonzero().flatten().tolist() + [positions.numel()]
        for start, end in zip(starts, starts[1:]):
            matches = pending.get(tuple(ids[start:end].tolist()))
            if matches:
                found[matches.pop(0)] = (step, start, end)
    missing = sorted(set(range(token_ids.shape[0])) - found.keys())
    if missing:
        raise RuntimeError(f"No captured activations for submitted sequences {missing}")
    return [found[index] for index in range(token_ids.shape[0])]


def assemble_rows(
    steps: Iterable[dict[str, torch.Tensor]], token_ids: torch.Tensor, slugs: list[str]
) -> dict[str, torch.Tensor]:
    """Per-site ``(sequences * seq_len, d)`` activations for ``token_ids``, in row order (see ``match_rows``)."""
    spans = match_rows(steps, token_ids)
    return {slug: torch.cat([step[slug][start:end] for step, start, end in spans]) for slug in slugs}


def _write_rows(rows: torch.Tensor, path: Path) -> float:
    """Write one site's rows of a fill; return when it finished, for the write rate."""
    save_file({"activations": rows}, str(path))
    return time.perf_counter()


def _arg(args: tuple, kwargs: dict, index: int, name: str):
    return kwargs[name] if name in kwargs else args[index]


def _residual(hidden: torch.Tensor, residual: torch.Tensor | None) -> torch.Tensor:
    # vLLM keeps the residual stream split as (mixer output, running residual).
    return hidden.clone() if residual is None else hidden + residual


class ActivationHookExtension:
    def _sae_model(self) -> nn.Module:
        # The only vLLM worker internal this extension relies on.
        return self.model_runner.model  # type: ignore[attr-defined]

    def sae_install_hooks(self, sites: list[tuple[str, str, int]], out_dir: str | None) -> None:
        """Hook every ``(slug, component kind, layer)``; steps go to files in ``out_dir``, or stay on the GPU."""
        model = self._sae_model()
        layers = model.model.layers
        self.sae_slugs = [slug for slug, _kind, _layer in sites]
        self.sae_out_dir = None if out_dir is None else Path(out_dir)
        if self.sae_out_dir is not None:
            self.sae_out_dir.mkdir(parents=True, exist_ok=True)
        self.sae_steps: list[dict[str, torch.Tensor]] = []
        self.sae_step = 0
        self.sae_captured: dict[str, torch.Tensor] = {}
        self.sae_inputs: dict[str, torch.Tensor] = {}
        self.sae_writes: list[tuple[Path, Future]] = []
        self.sae_writer = ThreadPoolExecutor(max_workers=1)
        self.sae_host: dict[str, torch.Tensor] = {}
        self.sae_save_pool: ThreadPoolExecutor | None = None
        self.sae_pending_save = None
        self.sae_handles = [
            model.register_forward_pre_hook(self._sae_record_inputs, with_kwargs=True),
            model.register_forward_hook(self._sae_end_step),
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

    def sae_join(self, host: str, port: int) -> None:
        """Join the trainer's activation group as the producer."""
        store = open_store(host, port, trainer=False)
        expected = json.loads(store.get(SITES_KEY))
        if expected["slugs"] != self.sae_slugs:
            raise ValueError(f"The trainer expects sites {expected['slugs']}, this producer hooks {self.sae_slugs}")
        self.sae_expected = dict(zip(expected["slugs"], expected["widths"])), expected["dtype"]
        self.sae_peer = peer_group(store, PRODUCER_RANK, next(self._sae_model().parameters()).device)

    def sae_send(self, token_ids: list[list[int]]) -> None:
        """Send the kept steps' rows for these prompts to the trainer, one site at a time."""
        prompts = torch.tensor(token_ids)
        widths, dtype = self.sae_expected
        for slug in self.sae_slugs:
            rows = assemble_rows(self.sae_steps, prompts, [slug])[slug]
            if rows.shape[1] != widths[slug] or str(rows.dtype) != dtype:
                raise ValueError(
                    f"{slug}: producing {rows.dtype} width {rows.shape[1]}, the trainer expects {dtype} width {widths[slug]}"
                )
            self.sae_peer.send([rows], TRAINER_RANK, 0).wait()
            for step in self.sae_steps:
                del step[slug]
        self.sae_steps = []

    def sae_save(
        self, token_ids: list[list[int]], out_dir: str, threads: int, extra_files: dict[str, bytes] | None = None
    ) -> dict[str, float] | None:
        """Copy the kept steps' rows for these prompts to host memory and write them as one fill directory.

        ``out_dir`` gets ``token_ids.safetensors``, one ``{slug}.safetensors``
        per site (key ``activations``), so a consumer reads only its sites, and
        ``extra_files`` as given. The rows go to host buffers reused across
        fills (pinned, so the copy runs at full speed) and are written under
        ``{out_dir}.partial`` on ``threads`` threads in the background, while
        the next fill is generated. The next ``sae_save`` (or
        ``sae_flush_saves``) waits for them and publishes the fill by renaming
        it, so a reader on another node never sees half a fill; it returns that
        fill's report, None for the first.
        """
        previous = self.sae_flush_saves()
        started = time.perf_counter()
        prompts = torch.tensor(token_ids)
        partial = Path(f"{out_dir}.partial")
        partial.mkdir(parents=True)
        save_file({"token_ids": prompts}, str(partial / "token_ids.safetensors"))
        for name, data in (extra_files or {}).items():
            (partial / name).write_bytes(data)
        nbytes = 0
        spans = match_rows(self.sae_steps, prompts)
        for slug in self.sae_slugs:
            # Each prompt's rows go straight to host memory: no assembled copy on the GPU.
            parts = [step[slug][start:end] for step, start, end in spans]
            shape = (sum(part.shape[0] for part in parts), parts[0].shape[1])
            host = self.sae_host.get(slug)
            if host is None or host.shape != shape or host.dtype != parts[0].dtype:
                host = torch.empty(shape, dtype=parts[0].dtype, pin_memory=torch.cuda.is_available())
                self.sae_host[slug] = host
            offset = 0
            for part in parts:
                host[offset : offset + part.shape[0]].copy_(part)
                offset += part.shape[0]
            nbytes += host.nbytes
            del parts
            for step in self.sae_steps:
                del step[slug]
        self.sae_steps = []
        copied = time.perf_counter()
        if self.sae_save_pool is None:
            self.sae_save_pool = ThreadPoolExecutor(max_workers=threads)
        writes = [
            self.sae_save_pool.submit(_write_rows, self.sae_host[slug], partial / f"{slug}.safetensors")
            for slug in self.sae_slugs
        ]
        self.sae_pending_save = (partial, Path(out_dir), writes, nbytes, copied - started, copied)
        return previous

    def sae_flush_saves(self) -> dict[str, float] | None:
        """Wait for the last ``sae_save``'s writes and publish its fill.

        Returns its bytes and the seconds spent copying them to host memory and
        writing them, or None if no fill was pending.
        """
        if self.sae_pending_save is None:
            return None
        partial, out_dir, writes, nbytes, copy_seconds, copied = self.sae_pending_save
        finished = max(write.result() for write in writes)
        partial.rename(out_dir)
        self.sae_pending_save = None
        return {"bytes": nbytes, "copy_seconds": copy_seconds, "write_seconds": finished - copied}

    def sae_vram(self) -> dict[str, int]:
        """Total VRAM and the least of it free since the last call (torch keeps what it reserved)."""
        free, total = torch.cuda.mem_get_info()
        lowest = free - (torch.cuda.max_memory_reserved() - torch.cuda.memory_reserved())
        torch.cuda.reset_peak_memory_stats()
        return {"min_free": lowest, "total": total}

    def sae_start_probe(self) -> None:
        """Start measuring a probe call's memory (see ``sae_end_probe``)."""
        torch.cuda.reset_peak_memory_stats()
        self.sae_probe_base = torch.cuda.memory_allocated()

    def sae_end_probe(self) -> dict[str, int]:
        """Drop the probe call's kept steps and report its memory, in bytes.

        ``free``: VRAM left once they are dropped; ``transient``: the peak above
        them while a step ran (activations, which later steps reuse);
        ``row_bytes``: what every kept row holds (captures and inputs);
        ``largest_site_row_bytes``: the extra copy ``assemble_rows`` makes of one site.
        """
        kept = sum(tensor.nbytes for step in self.sae_steps for tensor in step.values())
        rows = sum(step["input_ids"].numel() for step in self.sae_steps)
        largest = max(sum(step[slug].nbytes for step in self.sae_steps) for slug in self.sae_slugs)
        transient = torch.cuda.max_memory_allocated() - self.sae_probe_base - kept
        self.sae_steps = []
        torch.cuda.empty_cache()
        return {
            "free": torch.cuda.mem_get_info()[0],
            "transient": max(transient, 0),
            "row_bytes": kept // rows,
            "largest_site_row_bytes": largest // rows,
        }

    def sae_remove_hooks(self) -> None:
        self.sae_flush()
        self.sae_flush_saves()
        for handle in self.sae_handles:
            handle.remove()
        self.sae_writer.shutdown()
        if self.sae_save_pool is not None:
            self.sae_save_pool.shutdown()

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
        # Copy now: the next layer's fused add-RMSNorm overwrites this tensor in place.
        self.sae_captured[slug] = output.clone()

    def _sae_end_step(self, _module, _args, _output) -> None:
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return
        captured, self.sae_captured = self.sae_captured, {}
        if self.sae_out_dir is None:
            # vLLM reuses its input buffers for the next step.
            self.sae_steps.append({**{name: t.clone() for name, t in self.sae_inputs.items()}, **captured})
            return
        tensors = {**self.sae_inputs, **captured}
        cpu = {name: tensor.detach().to("cpu").contiguous() for name, tensor in tensors.items()}
        path = self.sae_out_dir / f"step_{self.sae_step:08d}.safetensors"
        self.sae_step += 1
        self.sae_writes.append((path, self.sae_writer.submit(save_file, cpu, str(path))))
