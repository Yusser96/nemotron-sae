"""vLLM extraction backend: native hidden-state extraction, written through a sink.

Uses vLLM's ``extract_hidden_states`` speculative method together with the
``ExampleHiddenStatesConnector``, which writes one safetensors file per
request into the sink directory. Each token sequence is one request, sent as
token ids (no re-tokenisation) with a single generated token. Only
residual-stream sites are available (see ``vllm_layout``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from sae_pipeline.config import ActivationSinkCfg, PipelineCfg, VLLMCfg
from sae_pipeline.extraction.base import ActivationBatch, UnsupportedComponentError
from sae_pipeline.extraction.sink import HiddenStateSink
from sae_pipeline.extraction.vllm_layout import VLLMLayout, residual_layer_id
from sae_pipeline.hooks.components import ComponentSpec
from sae_pipeline.sae.checkpoint import normalize_hf_source


def _vllm_engine(**engine_kwargs: Any) -> Any:
    from vllm import LLM
    from vllm.config.kv_transfer import KVTransferConfig

    engine_kwargs["kv_transfer_config"] = KVTransferConfig(**engine_kwargs["kv_transfer_config"])
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
        num_hidden_layers: int,
        seq_len: int,
        vllm_cfg: VLLMCfg,
        sink_cfg: ActivationSinkCfg,
        capture_device: str | torch.device = "cpu",
        engine_factory: Callable[..., Any] = _vllm_engine,
        sampling_params_factory: Callable[[], Any] = _one_token_sampling,
    ) -> None:
        self.num_hidden_layers = num_hidden_layers
        self.layout = VLLMLayout.from_specs(specs, num_hidden_layers)
        self.sink = HiddenStateSink(sink_cfg)
        self.capture_device = capture_device
        self.sequences_per_call = vllm_cfg.sequences_per_call
        self.engine_kwargs = {
            "model": model_name,
            "dtype": dtype,
            "trust_remote_code": trust_remote_code,
            "tensor_parallel_size": vllm_cfg.tensor_parallel_size,
            # Aux hidden-state ids are relative to the first local layer.
            "pipeline_parallel_size": 1,
            "gpu_memory_utilization": vllm_cfg.gpu_memory_utilization,
            "max_model_len": vllm_cfg.max_model_len or seq_len + 1,
            # Hidden-state extraction does not support chunked prefill.
            "enable_chunked_prefill": False,
            "enable_prefix_caching": vllm_cfg.enable_prefix_caching,
            "speculative_config": {
                "method": "extract_hidden_states",
                "num_speculative_tokens": 1,
                "draft_model_config": {
                    "hf_config": {"eagle_aux_hidden_state_layer_ids": list(self.layout.layer_ids)},
                },
            },
            "kv_transfer_config": {
                "kv_connector": "ExampleHiddenStatesConnector",
                "kv_role": "kv_producer",
                "kv_connector_extra_config": {
                    "shared_storage_path": str(self.sink.prepare()),
                    "num_writer_threads": sink_cfg.writer_threads,
                    "use_synchronization_lock": sink_cfg.synchronization_lock,
                },
            },
        }
        self.llm = engine_factory(**self.engine_kwargs)
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
        from transformers import AutoConfig

        model_name = normalize_hf_source(cfg.model.name)
        model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=cfg.model.trust_remote_code)
        return cls(
            model_name=model_name,
            dtype=cfg.model.dtype,
            trust_remote_code=cfg.model.trust_remote_code,
            specs=specs,
            num_hidden_layers=model_config.num_hidden_layers,
            seq_len=cfg.data.seq_len,
            vllm_cfg=cfg.extraction.vllm,
            sink_cfg=cfg.extraction.sink,
            capture_device=capture_device,
            **factories,
        )

    def supports(self, spec: ComponentSpec) -> bool:
        try:
            residual_layer_id(spec, self.num_hidden_layers)
        except (UnsupportedComponentError, ValueError):
            return False
        return True

    def extract(self, token_ids: torch.Tensor) -> ActivationBatch:
        rows = token_ids.detach().cpu()
        parts: dict[str, list[torch.Tensor]] = {slug: [] for slug in self.layout.columns}
        for start in range(0, rows.shape[0], self.sequences_per_call):
            chunk = rows[start : start + self.sequences_per_call]
            prompts = [{"prompt_token_ids": row.tolist()} for row in chunk]
            # generate() returns outputs in prompt order; the sink re-checks
            # every file's token ids against the row it is matched to.
            outputs = self.llm.generate(prompts, self.sampling_params, use_tqdm=False)
            if len(outputs) != len(prompts):
                raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(prompts)} prompts")
            for row, output in zip(chunk, outputs):
                activations = self.sink.read(
                    output.kv_transfer_params["hidden_states_path"],
                    expected_token_ids=row,
                    layout=self.layout,
                    device=self.capture_device,
                )
                for slug, value in activations.items():
                    parts[slug].append(value)
        return ActivationBatch(
            token_ids=token_ids,
            activations={slug: torch.cat(values, dim=0) for slug, values in parts.items()},
        )

    def close(self) -> None:
        self.llm = None
        self.sink.cleanup()
