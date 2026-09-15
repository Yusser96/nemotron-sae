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
CheckpointMode = Literal["finetune", "resume"]


class ModelCfg(BaseModel):
    name: str = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    load_quant: Quant = "auto"
    trust_remote_code: bool = True
    device_map: str = "auto"


class DataSourceCfg(BaseModel):
    """One independently streamed corpus in a multilingual activation mix."""

    source: str
    name: str | None = None
    text_field: str = "text"
    language: str | None = None


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
    # When present, these sources replace the legacy source/name fields.  Each
    # source contributes equally by token count.
    sources: list[DataSourceCfg] | None = None
    validation_fraction: float = 0.05
    validation_tokens_per_language: int | None = None

    @model_validator(mode="after")
    def _exactly_one_budget(self) -> "DataCfg":
        if (self.n_documents is None) == (self.total_tokens is None):
            raise ValueError("Set exactly one of n_documents or total_tokens.")
        if self.sources is not None and not self.sources:
            raise ValueError("sources must not be empty")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between zero and one")
        if self.validation_tokens_per_language is not None and self.validation_tokens_per_language <= 0:
            raise ValueError("validation_tokens_per_language must be positive")
        return self


class CacheCfg(BaseModel):
    shard_size_bytes: int = 1 << 30  # 1 GiB
    shuffle_seed: int = 42
    tokens_per_fwd: int = 8192
    cache_dir: Path = Path("outputs/caches")
    max_total_bytes: int | None = None


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
    checkpoint_source: str | Path | None = None
    checkpoint_mode: CheckpointMode = "finetune"
    checkpoint_filename: str | None = None
    checkpoint_revision: str = "main"
    keep_last_checkpoints: int = 2

    @field_validator("d_sae")
    @classmethod
    def _d_sae_positive(cls, v: int | list[int]) -> int | list[int]:
        widths = v if isinstance(v, list) else [v]
        for w in widths:
            if w <= 0:
                raise ValueError(f"d_sae must be positive, got {w}")
        return v

    @field_validator("keep_last_checkpoints")
    @classmethod
    def _keep_last_checkpoints_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("keep_last_checkpoints must be positive")
        return v


class HookTargetCfg(BaseModel):
    """A released SAE site, addressed by its exact model module path."""

    slug: str
    layer: int
    component: str
    hook_name: str


class TargetCfg(BaseModel):
    """What to hook. For prod, lists; CLI/launcher iterates the cartesian product."""
    layer: int | list[int] | None = None
    component: str | list[str] | None = None  # e.g. "resid_post", "moe_out", "expert.42"
    sites: list[HookTargetCfg] | None = None

    @model_validator(mode="after")
    def _targets_are_unambiguous(self) -> "TargetCfg":
        has_legacy = self.layer is not None or self.component is not None
        if has_legacy and (self.layer is None or self.component is None):
            raise ValueError("Set both target.layer and target.component")
        if self.sites is not None and has_legacy:
            raise ValueError("Use either target.sites or target.layer/component")
        if not has_legacy and not self.sites:
            raise ValueError("Set target.sites or target.layer/component")
        if self.sites is not None:
            slugs = [site.slug for site in self.sites]
            if len(slugs) != len(set(slugs)):
                raise ValueError("target.sites slugs must be unique")
        return self


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
