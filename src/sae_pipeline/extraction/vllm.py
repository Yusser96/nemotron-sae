"""vLLM extraction backend: forward hooks inside the vLLM workers.

The engine is started with ``vllm_worker.ActivationHookExtension`` as its
``worker_extension_cls``; hooks are installed once over ``collective_rpc``.
Each token sequence is one request, sent as token ids (no re-tokenisation)
with a single generated token; the workers write every forward pass's rows to
the sink, and the driver reassembles them per prompt.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from sae_pipeline.config import ActivationSinkCfg, PipelineCfg, VLLMCfg
from sae_pipeline.extraction.base import ActivationBatch
from sae_pipeline.extraction.sink import ActivationSink
from sae_pipeline.extraction.vllm_worker import hookable
from sae_pipeline.hooks.components import ComponentSpec
from sae_pipeline.sae.checkpoint import normalize_hf_source

WORKER_EXTENSION = "sae_pipeline.extraction.vllm_worker.ActivationHookExtension"


def _vllm_engine(**engine_kwargs: Any) -> Any:
    from vllm import LLM

    return LLM(**engine_kwargs)


def _one_token_sampling() -> Any:
    from vllm import SamplingParams

    return SamplingParams(max_tokens=1)


class VLLMExtractor:
    def __init__(
        self,
        *,
        model_name: str,
        dtype: str,
        trust_remote_code: bool,
        specs: dict[str, ComponentSpec],
        seq_len: int,
        vllm_cfg: VLLMCfg,
        sink_cfg: ActivationSinkCfg,
        capture_device: str | torch.device = "cpu",
        engine_factory: Callable[..., Any] = _vllm_engine,
        sampling_params_factory: Callable[[], Any] = _one_token_sampling,
    ) -> None:
        unsupported = [slug for slug, spec in specs.items() if not hookable(spec)]
        if unsupported:
            raise ValueError(f"vLLM extraction cannot hook {unsupported}")
        self.slugs = list(specs)
        self.sink = ActivationSink(sink_cfg)
        self.capture_device = capture_device
        self.sequences_per_call = vllm_cfg.sequences_per_call
        self.engine_kwargs = {
            "model": model_name,
            "dtype": dtype,
            "trust_remote_code": trust_remote_code,
            "tensor_parallel_size": vllm_cfg.tensor_parallel_size,
            "gpu_memory_utilization": vllm_cfg.gpu_memory_utilization,
            "max_model_len": vllm_cfg.max_model_len or seq_len + 1,
            # Whole prompts per forward pass, and all of one call's prompts in one batch.
            "enable_chunked_prefill": False,
            "max_num_batched_tokens": vllm_cfg.sequences_per_call * (seq_len + 1),
            "enable_prefix_caching": vllm_cfg.enable_prefix_caching,
            "enforce_eager": vllm_cfg.enforce_eager,
            "worker_extension_cls": WORKER_EXTENSION,
        }
        self.llm = engine_factory(**self.engine_kwargs)
        sites = [(slug, spec.kind, spec.layer) for slug, spec in specs.items()]
        self.llm.collective_rpc("sae_install_hooks", args=(sites, str(self.sink.prepare())))
        self.sampling_params = sampling_params_factory()

    @classmethod
    def from_config(
        cls,
        cfg: PipelineCfg,
        specs: dict[str, ComponentSpec],
        *,
        capture_device: str | torch.device = "cpu",
        **factories: Any,
    ) -> "VLLMExtractor":
        """Build from a pipeline config (model, ``data.seq_len``, ``extraction.vllm``/``sink``).

        ``factories`` forwards ``engine_factory`` / ``sampling_params_factory``.
        """
        return cls(
            model_name=normalize_hf_source(cfg.model.name),
            dtype=cfg.model.dtype,
            trust_remote_code=cfg.model.trust_remote_code,
            specs=specs,
            seq_len=cfg.data.seq_len,
            vllm_cfg=cfg.extraction.vllm,
            sink_cfg=cfg.extraction.sink,
            capture_device=capture_device,
            **factories,
        )

    def supports(self, spec: ComponentSpec) -> bool:
        return hookable(spec)

    def extract(self, token_ids: torch.Tensor) -> ActivationBatch:
        parts: dict[str, list[torch.Tensor]] = {slug: [] for slug in self.slugs}
        for chunk in token_ids.detach().cpu().split(self.sequences_per_call):
            prompts = [{"prompt_token_ids": row.tolist()} for row in chunk]
            self.llm.generate(prompts, self.sampling_params, use_tqdm=False)
            step_paths = [path for paths in self.llm.collective_rpc("sae_flush") for path in paths]
            activations = self.sink.collect(step_paths, chunk, self.slugs, device=self.capture_device)
            for slug, value in activations.items():
                parts[slug].append(value)
        return ActivationBatch(
            token_ids=token_ids,
            activations={slug: torch.cat(values, dim=0) for slug, values in parts.items()},
        )

    def close(self) -> None:
        self.llm.collective_rpc("sae_remove_hooks")
        self.llm = None
        self.sink.cleanup()
