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

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from sae_pipeline.config import PipelineCfg
from sae_pipeline.extraction.base import ActivationBatch
from sae_pipeline.extraction.remote import CALL_SIZE_KEY, open_store, take_tokens
from sae_pipeline.extraction.sink import ActivationSink
from sae_pipeline.extraction.vllm_worker import assemble_rows, hookable
from sae_pipeline.hooks.components import ComponentSpec
from sae_pipeline.sae.checkpoint import normalize_hf_source

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
    log.info("sequences_per_call: auto -> %d (%s, %.1f GiB headroom)", fits, memories, vllm_cfg.vram_headroom_gib)
    return fits


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
        self.sink = ActivationSink(cfg.extraction.sink)
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


def produce_fills(
    cfg: PipelineCfg,
    specs: dict[str, ComponentSpec],
    token_stream: TokenStream,
    *,
    out_dir: Path,
    fills: int,
    threads: int,
    engine_factory: Callable[..., Any] = _vllm_engine,
    sampling_params_factory: Callable[[], Any] = _one_token_sampling,
) -> list[dict[str, float]]:
    """Write ``fills`` fills of ``token_stream`` sequences to ``out_dir/fill_{i}`` (see ``sae_save``).

    Each fill is one ``generate`` call of ``sequences_per_call`` sequences whose
    captures stay in VRAM until they are written. Returns one report per fill:
    its size, the generate rate, the lowest free VRAM and the write rates.
    """
    llm = _start_engine(cfg, _hook_sites(specs), out_dir=None, engine_factory=engine_factory)
    sampling_params = sampling_params_factory()
    sequences = _sequences_per_call(llm, cfg, sampling_params)
    reports = []
    for index in range(fills):
        token_ids = token_stream.next_batch(sequences)
        llm.collective_rpc("sae_vram")
        started = time.perf_counter()
        llm.generate(_prompts(token_ids), sampling_params, use_tqdm=False)
        generate_seconds = time.perf_counter() - started
        vram = min(llm.collective_rpc("sae_vram"), key=lambda worker: worker["min_free"])
        save = llm.collective_rpc("sae_save", args=(token_ids.tolist(), str(out_dir / f"fill_{index:06d}"), threads))[0]
        report = {
            "fill": index,
            "sequences": sequences,
            "tokens": token_ids.numel(),
            "generate_seconds": generate_seconds,
            "tokens_per_s": token_ids.numel() / generate_seconds,
            "min_free_vram_gib": vram["min_free"] / 2**30,
            "total_vram_gib": vram["total"] / 2**30,
            "saved_gb": save["bytes"] / 1e9,
            "copy_gb_per_s": save["bytes"] / save["copy_seconds"] / 1e9,
            "save_gb_per_s": save["bytes"] / save["seconds"] / 1e9,
        }
        log.info("fill %d: %s", index, report)
        reports.append(report)
    llm.collective_rpc("sae_remove_hooks")
    return reports
