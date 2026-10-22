"""Load Nemotron-3-Nano-30B-A3B (or any HF causal LM) with the right dtype/quantization.

`mamba_ssm` and `causal-conv1d` are imported lazily by `transformers` via the model's
trust_remote_code modeling file when Mamba-2 layers are touched. We don't import them
ourselves; pip install supplies them at the system level.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast

from sae_pipeline.config import ModelCfg
from sae_pipeline.sae.checkpoint_source import (normalize_hf_source)

log = logging.getLogger(__name__)

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _normalise_time_step_limit(value: Any) -> Any:
    """Decode Transformers' JSON sentinel for an infinite Mamba time-step bound."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return value

    decoded = []
    for bound in value:
        if isinstance(bound, dict) and bound.get("__float__") == "Infinity":
            bound = float("inf")
        elif isinstance(bound, dict) and bound.get("__float__") == "-Infinity":
            bound = float("-inf")
        decoded.append(bound)
    if all(isinstance(bound, (int, float)) for bound in decoded):
        return tuple(float(bound) for bound in decoded)
    return value


def _normalise_model_runtime_config(model: torch.nn.Module) -> None:
    """Repair JSON-decoded Mamba limits before the first CUDA forward pass."""
    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "time_step_limit"):
        config.time_step_limit = _normalise_time_step_limit(config.time_step_limit)

    repaired = 0
    for module in model.modules():
        if hasattr(module, "time_step_limit"):
            normalised = _normalise_time_step_limit(module.time_step_limit)
            if normalised != module.time_step_limit:
                module.time_step_limit = normalised
                repaired += 1
    if repaired:
        log.info("Normalised Mamba time-step limits in %d mixer modules", repaired)


def _resolve_quantization_kwargs(cfg: ModelCfg) -> dict[str, Any]:
    """BF16, ``cfg.dtype`` (auto), or 4-bit NF4 weights; pre-quantized checkpoints load via auto."""
    if cfg.load_quant == "bf16":
        return {"torch_dtype": torch.bfloat16}

    if cfg.load_quant == "nf4":
        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        return {"quantization_config": bnb}

    return {"torch_dtype": _DTYPE_MAP[cfg.dtype]}


def _model_auth_kwargs(cfg: ModelCfg) -> dict[str, str]:
    if cfg.token_env_var is None:
        return {}
    value = os.environ.get(cfg.token_env_var)
    if not value:
        raise ValueError(f'Model credential environment variable is missing: {cfg.token_env_var}')
    return {'token': value}


def load_tokenizer(cfg: ModelCfg):
    model_source = normalize_hf_source(cfg.name)
    log.info("Loading tokenizer for %s", model_source)
    kwargs = dict(trust_remote_code=cfg.trust_remote_code, **_model_auth_kwargs(cfg))
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_source, **kwargs)
    except ValueError as exc:
        # Transformers 5 renamed the generic Rust tokenizer backend. A complete
        # tokenizer.json remains loadable by the corresponding v4 base class.
        if str(exc) != "Tokenizer class TokenizersBackend does not exist or is not currently imported.":
            raise
        log.warning("Loading Transformers 5 TokenizersBackend with the compatible fast tokenizer")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_source, **kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def load_model_and_tokenizer(cfg: ModelCfg):
    """Load the LM and its tokenizer. The model is left in eval mode; weights are frozen
    (we never train the LM, only SAEs that read its activations)."""
    quant_kwargs = _resolve_quantization_kwargs(cfg)
    model_source = normalize_hf_source(cfg.name)
    tokenizer = load_tokenizer(cfg)

    log.info("Loading model %s with %s", model_source, quant_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        device_map=cfg.device_map,
        trust_remote_code=cfg.trust_remote_code,
        **quant_kwargs,
        **_model_auth_kwargs(cfg),
    )
    _normalise_model_runtime_config(model)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return model, tokenizer
