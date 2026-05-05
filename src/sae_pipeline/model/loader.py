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

log = logging.getLogger(__name__)

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


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

    log.info("Loading tokenizer for %s", cfg.name)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.name, trust_remote_code=cfg.trust_remote_code
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    log.info("Loading model %s with %s", cfg.name, quant_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.name,
        device_map=cfg.device_map,
        trust_remote_code=cfg.trust_remote_code,
        **quant_kwargs,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return model, tokenizer
