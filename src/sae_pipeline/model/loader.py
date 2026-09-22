"""Load Nemotron-3-Nano-30B-A3B (or any HF causal LM) with the right dtype/quantization.

`mamba_ssm` and `causal-conv1d` are imported lazily by `transformers` via the model's
trust_remote_code modeling file when Mamba-2 layers are touched. We don't import them
ourselves; pip install supplies them at the system level.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sae_pipeline.config import ModelCfg
from sae_pipeline.sae.checkpoint import normalize_hf_source

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
    """Decide whether to apply BF16, FP8 (sibling repo), or 4-bit quantization."""
    if cfg.load_quant == "bf16":
        return {"torch_dtype": torch.bfloat16}

    if cfg.load_quant == "fp8":
        # Use the FP8 sibling repo instead of mid-flight quantization.
        log.info("FP8 selected: caller should set model.name to the FP8 sibling repo.")
        return {"torch_dtype": torch.bfloat16}  # FP8 weights load in BF16-typed wrapper

    if cfg.load_quant == "nf4":
        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        return {"quantization_config": bnb}

    # auto: use BF16 by default
    return {"torch_dtype": _DTYPE_MAP[cfg.dtype]}


def load_model_and_tokenizer(cfg: ModelCfg):
    """Load the LM and its tokenizer. The model is left in eval mode; weights are frozen
    (we never train the LM, only SAEs that read its activations)."""
    quant_kwargs = _resolve_quantization_kwargs(cfg)
    model_source = normalize_hf_source(cfg.name)

    log.info("Loading tokenizer for %s", model_source)
    tokenizer = AutoTokenizer.from_pretrained(
        model_source, trust_remote_code=cfg.trust_remote_code
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    log.info("Loading model %s with %s", model_source, quant_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        device_map=cfg.device_map,
        trust_remote_code=cfg.trust_remote_code,
        **quant_kwargs,
    )
    _normalise_model_runtime_config(model)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return model, tokenizer
