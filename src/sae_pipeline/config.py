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
# Only JumpReLU is implemented in the current trainer.  Keep unsupported
# architectures out of the public configuration schema until their trainers,
# checkpoint semantics and evaluation paths exist.
SAEArch = Literal["jumprelu"]
CheckpointMode = Literal["finetune", "resume"]
InputNormalization = Literal["none", "whole_vector"]
CheckpointFormat = Literal["native", "raw_export"]
LRSchedule = Literal["cosine_decay", "warmup_constant"]
ReconstructionLoss = Literal["coordinate_mean", "vector_sum"]
FeatureUseStrategy = Literal[
    "none",
    "frequency",
    "residual_reset",
    "frequency_residual_reset",
]
ActivationCentering = Literal["none", "mean"]


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
    input_normalization: InputNormalization = "none"
    checkpoint_format: CheckpointFormat = "native"
    lr_schedule: LRSchedule = "cosine_decay"
    gradient_clip_norm: float | None = None
    normalization_dir: Path | None = None
    # ``vector_sum`` matches the per-token ||x - x_hat||_2^2 objective used by
    # the Gemma Scope recipe.  ``coordinate_mean`` is retained for
    # compatibility with the original trainer.
    reconstruction_loss: ReconstructionLoss = "coordinate_mean"
    l0_penalty_scale: float = 1.0
    seed: int = 42
    l0_target_start: float | None = None
    l0_target_warmup_steps: int = 0
    decoder_freeze_steps: int = 0
    activation_centering: ActivationCentering = "none"
    centering_sample_tokens: int = 1_000_000
    centering_dir: Path | None = None
    active_subspace_rank: int | None = None
    active_subspace_sample_tokens: int = 100_000
    # Optional controls for the long repair sweep.  They are deliberately
    # explicit in the config so an intervention is reproducible and auditable.
    feature_use_strategy: FeatureUseStrategy = "none"
    feature_frequency_start_step: int = 50_000
    feature_frequency_ema_decay: float = 0.999
    feature_frequency_threshold: float = 0.05
    feature_frequency_penalty_fraction: float = 0.1
    residual_reset_start_step: int = 50_000
    residual_reset_every_steps: int = 30_000
    residual_reset_max_features: int = 512
    residual_reset_frequency_threshold: float = 1.0e-6
    residual_reset_pool_size: int = 8_192
    residual_reset_target_frequency: float = 1.0e-3
    residual_reset_delta_l0: float = 2.0
    residual_reset_calibration_tokens: int = 1_048_576
    residual_reset_anneal_steps: int = 10_000
    feature_frequency_penalty_multiplier: float = 1.0
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

    @field_validator("gradient_clip_norm")
    @classmethod
    def _gradient_clip_norm_positive(cls, v: float | None) -> float | None:
        if v is not None and v <= 0:
            raise ValueError("gradient_clip_norm must be positive when set")
        return v

    @field_validator(
        "l0_penalty_scale",
        "feature_frequency_penalty_fraction",
        "residual_reset_delta_l0",
    )
    @classmethod
    def _positive_scales(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("scale values must be positive")
        return v

    @field_validator("feature_frequency_penalty_multiplier")
    @classmethod
    def _frequency_multiplier_nonnegative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("feature_frequency_penalty_multiplier must be non-negative")
        return v

    @field_validator("feature_frequency_ema_decay")
    @classmethod
    def _ema_decay_valid(cls, v: float) -> float:
        if not 0.0 <= v < 1.0:
            raise ValueError("feature_frequency_ema_decay must be in [0, 1)")
        return v

    @field_validator("feature_frequency_threshold", "residual_reset_target_frequency")
    @classmethod
    def _frequency_valid(cls, v: float) -> float:
        if not 0.0 < v <= 1.0:
            raise ValueError("feature frequency thresholds must be in (0, 1]")
        return v

    @field_validator(
        "feature_frequency_start_step",
        "residual_reset_start_step",
        "residual_reset_every_steps",
        "residual_reset_max_features",
        "residual_reset_pool_size",
    )
    @classmethod
    def _intervention_counts_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("intervention step and count values must be positive")
        return v

    @field_validator("l0_target_warmup_steps", "decoder_freeze_steps")
    @classmethod
    def _optional_intervention_steps_nonnegative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("intervention step values must be non-negative")
        return v

    @field_validator(
        "centering_sample_tokens",
        "active_subspace_sample_tokens",
        "residual_reset_calibration_tokens",
        "residual_reset_anneal_steps",
    )
    @classmethod
    def _sample_counts_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("diagnostic sample counts must be positive")
        return v

    @field_validator("active_subspace_rank")
    @classmethod
    def _subspace_rank_valid(cls, v: int | None) -> int | None:
        if v is not None and v <= 0:
            raise ValueError("active_subspace_rank must be positive when set")
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
    # Training and validation activation caches normally live under the same
    # run_id.  Some programmes train from one cache but validate against a
    # separately generated, larger diagnostic cache; set validation_run_id to
    # point the validation partition at that other run_id while training,
    # checkpoints and logs keep using run_id.
    validation_run_id: str | None = None
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
