"""Pydantic-typed configuration for the SAE pipeline.

Loaded from YAML; CLI flags overlay specific fields. One config feeds a sweep over
(layer, component, sae_arch, dict_width, l0_target).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


Quant = Literal["auto", "bf16", "fp8", "nf4"]
SAEArch = Literal["jumprelu", "topk", "batchtopk", "matryoshka"]


class ModelCfg(BaseModel):
    name: str = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    load_quant: Quant = "auto"
    trust_remote_code: bool = True
    device_map: str = "auto"


class DataCfg(BaseModel):
    source: str = "HuggingFaceFW/fineweb-edu"
    name: str | None = "sample-10BT"   # HF dataset config / subset name; None = default
    split: str = "train"
    streaming: bool = True
    n_documents: int | None = None  # dev: 100; prod: None means use total_tokens
    total_tokens: int | None = None
    seq_len: int = 1024
    text_field: str = "text"
    shuffle_buffer: int = 1000
    seed: int = 42

    @model_validator(mode="after")
    def _exactly_one_budget(self) -> "DataCfg":
        if (self.n_documents is None) == (self.total_tokens is None):
            raise ValueError("Set exactly one of n_documents or total_tokens.")
        return self


class CacheCfg(BaseModel):
    shard_size_bytes: int = 1 << 30  # 1 GiB
    shuffle_seed: int = 42
    tokens_per_fwd: int = 8192
    cache_dir: Path = Path("outputs/caches")


class SAECfg(BaseModel):
    arch: SAEArch = "jumprelu"
    d_sae: int | list[int] = 16384
    l0_target: int | list[int] = 50
    lr: float = 7.0e-5
    batch_size: int = 4096
    n_steps: int = 200_000
    warmup_steps: int = 1_000
    l0_warmup_steps: int = 50_000
    bandwidth: float = 0.001  # JumpReLU STE bandwidth
    adam_beta1: float = 0.0   # Gemma Scope 2 default
    adam_beta2: float = 0.999
    adam_eps: float = 1.0e-8
    decoder_unit_norm: bool = True
    pre_encoder_bias: bool = True
    dead_freq_threshold: float = 0.1  # direct frequency penalization on >10% latents
    n_batches_in_buffer: int = 8
    ckpt_every: int = 5_000
    log_every: int = 100
    ckpt_dir: Path = Path("outputs/checkpoints")

    @field_validator("d_sae")
    @classmethod
    def _d_sae_positive(cls, v: int | list[int]) -> int | list[int]:
        widths = v if isinstance(v, list) else [v]
        for w in widths:
            if w <= 0:
                raise ValueError(f"d_sae must be positive, got {w}")
        return v


class TargetCfg(BaseModel):
    """What to hook. For prod, lists; CLI/launcher iterates the cartesian product."""
    layer: int | list[int]
    component: str | list[str]  # e.g. "resid_post", "moe_out", "expert.42"


class EvalCfg(BaseModel):
    delta_ce_n_seqs: int = 2048
    delta_ce_seq_len: int = 1024
    fvu_n_tokens: int = 65_536
    dead_n_tokens: int = 50_000
    interp_enabled: bool = False  # auto-interp via OPENAI_API_KEY


class LogCfg(BaseModel):
    use_wandb: bool = False
    wandb_project: str = "nemotron-sae"
    log_dir: Path = Path("outputs/logs")


class PipelineCfg(BaseModel):
    run_id: str
    model: ModelCfg = Field(default_factory=ModelCfg)
    data: DataCfg
    cache: CacheCfg = Field(default_factory=CacheCfg)
    sae: SAECfg = Field(default_factory=SAECfg)
    target: TargetCfg
    eval: EvalCfg = Field(default_factory=EvalCfg)
    log: LogCfg = Field(default_factory=LogCfg)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineCfg":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def with_overrides(self, **overrides) -> "PipelineCfg":
        """Create a copy with specified fields overridden (used by CLI flags)."""
        as_dict = self.model_dump()
        for k, v in overrides.items():
            if v is None:
                continue
            # Dotted paths like "sae.d_sae" overlay nested fields.
            parts = k.split(".")
            target = as_dict
            for p in parts[:-1]:
                target = target[p]
            target[parts[-1]] = v
        return PipelineCfg(**as_dict)
