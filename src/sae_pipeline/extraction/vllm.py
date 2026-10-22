"""vLLM extraction backend: forward hooks inside the vLLM workers.

The engine is started with ``vllm_worker.ActivationHookExtension`` as its
``worker_extension_cls``; hooks are installed once over ``collective_rpc``.
Each token sequence is one request, sent as token ids (no re-tokenisation)
with a single generated token. ``VLLMExtractor`` has the workers write every
forward pass's rows to the sink and reassembles them per prompt;
``serve_activations`` instead answers a trainer in another process, with the
workers sending the rows to it GPU to GPU (see ``remote``); ``produce_fills``
has them write VRAM-sized fills, one file per site, for consumers elsewhere.
"""

from __future__ import annotations

import io
import logging
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from safetensors.torch import load_file

from sae_pipeline.config import ActivationSinkCfg, PipelineCfg
from sae_pipeline.extraction.base import ActivationBatch
from sae_pipeline.extraction.fills import FillQueue, fill_dir, published_fills, write_header
from sae_pipeline.extraction.remote import CALL_SIZE_KEY, open_store, take_tokens
from sae_pipeline.extraction.vllm_worker import assemble_rows, hookable
from sae_pipeline.hooks.components import ComponentSpec
from sae_pipeline.sae.checkpoint import (load_trainer_state)
from sae_pipeline.sae.checkpoint_source import (normalize_hf_source)

if TYPE_CHECKING:
    from sae_pipeline.sae.train_streaming import TokenStream

WORKER_EXTENSION = "sae_pipeline.extraction.vllm_worker.ActivationHookExtension"

log = logging.getLogger(__name__)


def _vllm_engine(**engine_kwargs: Any) -> Any:
    from vllm import LLM

    return LLM(**engine_kwargs)


def _one_token_sampling() -> Any:
    from vllm import SamplingParams

    return SamplingParams(max_tokens=1)


def _hook_sites(specs: dict[str, ComponentSpec]) -> list[tuple[str, str, int]]:
    """``(slug, component kind, layer)`` for the worker; rejects sites vLLM cannot hook."""
    unsupported = [slug for slug, spec in specs.items() if not hookable(spec)]
    if unsupported:
        raise ValueError(f"vLLM extraction cannot hook {unsupported}")
    return [(slug, spec.kind, spec.layer) for slug, spec in specs.items()]


def _start_engine(
    cfg: PipelineCfg,
    sites: list[tuple[str, str, int]],
    *,
    out_dir: str | None,
    engine_factory: Callable[..., Any] = _vllm_engine,
) -> Any:
    """Start vLLM for ``cfg.model`` with ``sites`` hooked (see ``sae_install_hooks``)."""
    vllm_cfg, seq_len = cfg.extraction.vllm, cfg.data.seq_len
    if vllm_cfg.tensor_parallel_size != 1 and any(kind == "attn_out_prelinear" for _, kind, _ in sites):
        raise ValueError("attn_out_prelinear requires tensor_parallel_size=1; projection inputs are sharded across ranks")
    llm = engine_factory(
        model=normalize_hf_source(cfg.model.name),
        dtype=cfg.model.dtype,
        trust_remote_code=cfg.model.trust_remote_code,
        tensor_parallel_size=vllm_cfg.tensor_parallel_size,
        gpu_memory_utilization=vllm_cfg.gpu_memory_utilization,
        max_model_len=vllm_cfg.max_model_len or seq_len + 1,
        # Whole prompts per forward pass, up to max_sequences_per_step of them.
        enable_chunked_prefill=False,
        max_num_batched_tokens=(vllm_cfg.max_sequences_per_step or vllm_cfg.sequences_per_call) * (seq_len + 1),
        kv_cache_memory_bytes=None if vllm_cfg.kv_cache_gib is None else int(vllm_cfg.kv_cache_gib * 2**30),
        enable_prefix_caching=vllm_cfg.enable_prefix_caching,
        enforce_eager=vllm_cfg.enforce_eager,
        worker_extension_cls=WORKER_EXTENSION,
    )
    llm.collective_rpc("sae_install_hooks", args=(sites, out_dir))
    return llm


def _prompts(token_ids: torch.Tensor) -> list[dict[str, list[int]]]:
    return [{"prompt_token_ids": row.tolist()} for row in token_ids]


def fit_sequences(memory: dict[str, int], *, seq_len: int, headroom_bytes: int) -> int:
    """How many sequences' captures fit in one worker's VRAM (``sae_end_probe``'s report)."""
    per_sequence = (memory["row_bytes"] + memory["largest_site_row_bytes"]) * seq_len
    fits = (memory["free"] - memory["transient"] - headroom_bytes) // per_sequence
    if fits < 1:
        raise RuntimeError(f"Not even one sequence's captures fit next to the engine and the headroom: {memory}")
    return fits


def _sequences_per_call(llm: Any, cfg: PipelineCfg, sampling_params: Any) -> int:
    """``extraction.vllm.sequences_per_call``, fitted to the free VRAM when it is ``auto``."""
    vllm_cfg, seq_len = cfg.extraction.vllm, cfg.data.seq_len
    if vllm_cfg.sequences_per_call != "auto":
        return vllm_cfg.sequences_per_call
    # One full step of dummy ids (as in vLLM's own profiling run), so the probe
    # sees the activation peak every later step reaches.
    probe = torch.zeros(vllm_cfg.max_sequences_per_step, seq_len, dtype=torch.long)
    llm.collective_rpc("sae_start_probe")
    llm.generate(_prompts(probe), sampling_params, use_tqdm=False)
    memories = llm.collective_rpc("sae_end_probe")
    headroom = int(vllm_cfg.vram_headroom_gib * 2**30)
    fits = min(fit_sequences(memory, seq_len=seq_len, headroom_bytes=headroom) for memory in memories)
    if vllm_cfg.max_sequences_per_call is not None:
        fits = min(fits, vllm_cfg.max_sequences_per_call)
    log.info("sequences_per_call: auto -> %d (%s, %.1f GiB headroom)", fits, memories, vllm_cfg.vram_headroom_gib)
    return fits


class _ActivationSink:
    """The step files the worker hooks write to ``extraction.sink``; ``consume`` deletes each once read."""

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


class VLLMExtractor:
    def __init__(
        self,
        cfg: PipelineCfg,
        specs: dict[str, ComponentSpec],
        *,
        capture_device: str | torch.device = "cpu",
        engine_factory: Callable[..., Any] = _vllm_engine,
        sampling_params_factory: Callable[[], Any] = _one_token_sampling,
    ) -> None:
        """Engine settings come from ``cfg.model``, ``data.seq_len`` and ``extraction.vllm``/``sink``."""
        sites = _hook_sites(specs)
        if cfg.extraction.vllm.sequences_per_call == "auto":
            raise ValueError("sequences_per_call: auto sizes captures kept on the GPU; the file sink needs a number")
        self.slugs = list(specs)
        self.sink = _ActivationSink(cfg.extraction.sink)
        self.capture_device = capture_device
        self.sequences_per_call = cfg.extraction.vllm.sequences_per_call
        self.llm = _start_engine(cfg, sites, out_dir=str(self.sink.prepare()), engine_factory=engine_factory)
        self.sampling_params = sampling_params_factory()

    def extract(self, token_ids: torch.Tensor) -> ActivationBatch:
        parts: dict[str, list[torch.Tensor]] = {slug: [] for slug in self.slugs}
        for chunk in token_ids.detach().cpu().split(self.sequences_per_call):
            self.llm.generate(_prompts(chunk), self.sampling_params, use_tqdm=False)
            step_paths = [path for paths in self.llm.collective_rpc("sae_flush") for path in paths]
            activations = assemble_rows(self.sink.read(step_paths), chunk, self.slugs)
            for slug, value in activations.items():
                parts[slug].append(value.to(self.capture_device))
        return ActivationBatch(
            token_ids=token_ids,
            activations={slug: torch.cat(values, dim=0) for slug, values in parts.items()},
        )

    def close(self) -> None:
        self.llm.collective_rpc("sae_remove_hooks")
        self.llm = None
        self.sink.cleanup()


def serve_activations(
    cfg: PipelineCfg,
    specs: dict[str, ComponentSpec],
    *,
    host: str,
    port: int,
    engine_factory: Callable[..., Any] = _vllm_engine,
    sampling_params_factory: Callable[[], Any] = _one_token_sampling,
) -> int:
    """Answer a ``remote.RemoteExtractor``'s token batches until it stops; return how many."""
    llm = _start_engine(cfg, _hook_sites(specs), out_dir=None, engine_factory=engine_factory)
    sampling_params = sampling_params_factory()
    sequences_per_call = _sequences_per_call(llm, cfg, sampling_params)
    store = open_store(host, port, trainer=False)
    store.set(CALL_SIZE_KEY, str(sequences_per_call))
    llm.collective_rpc("sae_join", args=(host, port))
    served = 0
    while (token_ids := take_tokens(store, served)) is not None:
        for chunk in token_ids.split(sequences_per_call):
            llm.generate(_prompts(chunk), sampling_params, use_tqdm=False)
            llm.collective_rpc("sae_send", args=(chunk.tolist(),))
        served += 1
    llm.collective_rpc("sae_remove_hooks")
    return served


def _resume(out_dir: Path, token_stream: TokenStream, start_state: Path | None) -> int:
    """Drop half-written fills, resume ``token_stream`` after the newest fill; return the next fill's index."""
    for partial in out_dir.glob("fill_*.partial"):
        shutil.rmtree(partial)
    published = published_fills(out_dir)
    state_path = fill_dir(out_dir, published[-1]) / "stream_state.pt" if published else start_state
    if state_path is not None:
        token_stream.load_state_dict(load_trainer_state(state_path)["token_stream"])
        log.info("Resumed the token stream from %s", state_path)
    return published[-1] + 1 if published else 0


def produce_fills(
    cfg: PipelineCfg,
    specs: dict[str, ComponentSpec],
    token_stream: TokenStream,
    *,
    out_dir: Path,
    threads: int,
    fills: int | None = None,
    queue: FillQueue | None = None,
    max_buffer_bytes: int | None = None,
    start_state: Path | None = None,
    stop_after_seconds: float | None = None,
    max_idle_seconds: float | None = None,
    engine_factory: Callable[..., Any] = _vllm_engine,
    sampling_params_factory: Callable[[], Any] = _one_token_sampling,
) -> list[dict[str, float]]:
    """Write fills of ``token_stream`` sequences to ``out_dir/fill_{i}`` (see ``sae_save`` and ``fills``).

    Each fill is one ``generate`` call of ``sequences_per_call`` sequences whose
    captures stay in VRAM until they are written, with the stream's state
    after it. A rerun continues after the newest fill in ``out_dir``
    (``start_state``, a saved token stream, starts an empty one). Stops after
    ``fills`` fills, after ``stop_after_seconds``, once every consumer of
    ``queue`` is done, or when ``max_buffer_bytes`` of fills have awaited them
    for ``max_idle_seconds`` (they have stalled: the GPU goes back).
    Returns one report per fill: its size, the generate rate, the lowest free
    VRAM, the write rates and the wait for room.
    """
    deadline = None if stop_after_seconds is None else time.monotonic() + stop_after_seconds
    first = _resume(out_dir, token_stream, start_state)
    write_header(out_dir, {"model": cfg.model.name, "seq_len": cfg.data.seq_len, "slugs": list(specs)})
    llm = _start_engine(cfg, _hook_sites(specs), out_dir=None, engine_factory=engine_factory)
    sampling_params = sampling_params_factory()
    sequences = _sequences_per_call(llm, cfg, sampling_params)
    reports = []
    index = first
    while fills is None or index - first < fills:
        waited = time.perf_counter()
        if queue is not None:
            if queue.buffered_bytes() >= max_buffer_bytes:
                # Consumers may be waiting for the fill still being written.
                _record_save(llm.collective_rpc("sae_flush_saves")[0], reports)
            limits = [deadline] if deadline is not None else []
            if max_idle_seconds is not None:
                limits.append(time.monotonic() + max_idle_seconds)
            if not queue.wait_for_room(max_buffer_bytes, deadline=min(limits, default=None)):
                log.info("Stopping: %s", "every consumer is done" if queue.all_done() else "no room for new fills")
                break
        waited = time.perf_counter() - waited
        if deadline is not None and time.monotonic() >= deadline:
            break
        cycle_started = time.perf_counter()
        token_ids = token_stream.next_batch(sequences)
        state = io.BytesIO()
        torch.save({"token_stream": token_stream.state_dict()}, state)
        llm.collective_rpc("sae_vram")
        started = time.perf_counter()
        llm.generate(_prompts(token_ids), sampling_params, use_tqdm=False)
        generate_seconds = time.perf_counter() - started
        vram = min(llm.collective_rpc("sae_vram"), key=lambda worker: worker["min_free"])
        # Returns the previous fill's save, written while this one was generated.
        saved = llm.collective_rpc(
            "sae_save",
            args=(token_ids.tolist(), str(fill_dir(out_dir, index)), threads, {"stream_state.pt": state.getvalue()}),
        )[0]
        _record_save(saved, reports)
        report = {
            "fill": index,
            "sequences": sequences,
            "tokens": token_ids.numel(),
            "waited_for_room_seconds": waited,
            "generate_seconds": generate_seconds,
            "tokens_per_s": token_ids.numel() / generate_seconds,
            # Generating plus copying out; the writes overlap the next fill.
            "cycle_tokens_per_s": token_ids.numel() / (time.perf_counter() - cycle_started),
            "min_free_vram_gib": vram["min_free"] / 2**30,
            "total_vram_gib": vram["total"] / 2**30,
        }
        log.info("fill %d: %s", index, report)
        reports.append(report)
        index += 1
    _record_save(llm.collective_rpc("sae_flush_saves")[0], reports)
    llm.collective_rpc("sae_remove_hooks")
    return reports


def _record_save(saved: dict[str, float] | None, reports: list[dict[str, float]]) -> None:
    """Add a published fill's write rates to its report."""
    if saved is None:
        return
    rates = {
        "saved_gb": saved["bytes"] / 1e9,
        "copy_gb_per_s": saved["bytes"] / saved["copy_seconds"] / 1e9,
        "write_gb_per_s": saved["bytes"] / saved["write_seconds"] / 1e9,
    }
    reports[-1].update(rates)
    log.info("fill %d published: %s", reports[-1]["fill"], rates)
