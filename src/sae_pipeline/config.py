"""Pydantic-typed configuration for the SAE pipeline.

Loaded from YAML; CLI flags overlay specific fields. One config feeds a sweep over
(layer, component, sae_arch, dict_width, l0_target).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


Quant = Literal["auto", "bf16", "fp8", "nf4"]
# Only JumpReLU has a trainer.
SAEArch = Literal["jumprelu"]
CheckpointMode = Literal["finetune", "resume"]
InputNormalization = Literal["none", "whole_vector"]
CheckpointFormat = Literal["native", "raw_export"]
LRSchedule = Literal["cosine_decay", "warmup_constant", "cosine_warmup_constant"]
ReconstructionLoss = Literal["coordinate_mean", "vector_sum"]
FeatureUseStrategy = Literal[
    "none",
    "frequency",
    "residual_reset",
    "frequency_residual_reset",
]
ActivationCentering = Literal["none", "mean"]
ExtractionBackend = Literal["hf_hooks", "vllm", "vllm_remote"]
SinkLifecycle = Literal["consume", "cache"]


class ModelCfg(BaseModel):
    name: str = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    load_quant: Quant = "auto"
    trust_remote_code: bool = True
    # A string ("auto", "balanced", ...) lets accelerate place the model across
    # every visible GPU; a mapping such as {"": "cuda:0"} pins it to one device.
    device_map: str | dict[str, str] = "auto"


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
    # ``vector_sum``: per-token squared L2 error, as in Gemma Scope.
    # ``coordinate_mean``: squared error averaged over coordinates.
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
    inverse_simpson_floor: float = 0.0
    inverse_simpson_max_scale: float = 0.0
    # None follows l0_warmup_steps so both firing-pattern terms ramp together.
    inverse_simpson_warmup_steps: int | None = None
    # Batches the firing rates behind the floor are averaged over (an EMA);
    # 1: each batch on its own. Longer windows apply to streaming training.
    inverse_simpson_window_steps: int = 1
    n_batches_in_buffer: int = 8
    ckpt_every: int = 5_000
    # Fixed-probe diagnostics cadence. Validation-based stopping has its own
    # minimum-step and patience settings.
    eval_interval_steps: int | None = 10_000
    log_every: int = 100
    ckpt_dir: Path = Path("outputs/checkpoints")
    checkpoint_source: str | Path | None = None
    checkpoint_mode: CheckpointMode = "finetune"
    checkpoint_filename: str | None = None
    # For multi-site runs: ``{slug}`` is replaced by each site's slug to pick
    # that site's file in checkpoint_source, e.g. "{slug}/sae_weights.safetensors".
    checkpoint_filename_template: str | None = None
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

    @field_validator("feature_frequency_penalty_multiplier", "inverse_simpson_max_scale")
    @classmethod
    def _frequency_multiplier_nonnegative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Multiplier and regularizer scales must be non-negative")
        return v

    @field_validator("inverse_simpson_floor")
    @classmethod
    def _inverse_simpson_floor_valid(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("inverse_simpson_floor must be in [0, 1]")
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
        "inverse_simpson_window_steps",
    )
    @classmethod
    def _intervention_counts_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("intervention step and count values must be positive")
        return v

    @field_validator(
        "l0_target_warmup_steps", "decoder_freeze_steps", "inverse_simpson_warmup_steps"
    )
    @classmethod
    def _optional_intervention_steps_nonnegative(cls, v: int | None) -> int | None:
        if v is not None and v < 0:
            raise ValueError("intervention step values must be non-negative")
        return v

    @field_validator("eval_interval_steps")
    @classmethod
    def _eval_interval_steps_positive(cls, v: int | None) -> int | None:
        if v is not None and v <= 0:
            raise ValueError("eval_interval_steps must be positive when set")
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
    # Training trajectories use one fixed sample for every metric and split.
    trajectory_n_tokens: int = 50_000
    interp_enabled: bool = False  # auto-interp via OPENAI_API_KEY

    @field_validator("fvu_n_tokens", "dead_n_tokens", "trajectory_n_tokens")
    @classmethod
    def _evaluation_token_counts_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("evaluation token counts must be positive")
        return value


EarlyStoppingMode = Literal["stability", "objective"]


class EarlyStoppingCfg(BaseModel):
    """Validation-based stopping for fine-tuning runs.

    ``mode="stability"`` stops once validation FVU/L0 stop moving between
    consecutive evaluations. ``mode="objective"`` is for long open-ended runs:
    it tracks the training loss evaluated on held-out data, reduces the
    learning rate when that plateaus, and stops after sustained
    non-improvement.
    """

    enabled: bool = False
    evaluation_interval_steps: int = 10_000
    minimum_steps: int = 50_000
    patience: int = 2
    fvu_relative_tolerance: float = 0.01
    ev_absolute_tolerance: float = 0.005
    l0_relative_tolerance: float = 0.10
    mode: EarlyStoppingMode = "stability"
    # ``objective`` mode only: an evaluation improves on the best so far when
    # it is lower by more than this relative margin; every
    # ``lr_plateau_patience`` non-improving evaluations scale the learning
    # rate by ``lr_reduce_factor`` (floored at ``min_lr_scale``), and
    # ``patience`` non-improving evaluations stop the run.
    improvement_rel_tol: float = 0.005
    lr_plateau_patience: int = 3
    lr_reduce_factor: float = 0.5
    min_lr_scale: float = 0.05

    @field_validator("evaluation_interval_steps", "patience", "lr_plateau_patience")
    @classmethod
    def _positive_early_stop_counts(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("early-stopping intervals and patience must be positive")
        return value

    @field_validator("minimum_steps")
    @classmethod
    def _nonnegative_early_stop_minimum(cls, value: int) -> int:
        if value < 0:
            raise ValueError("early-stopping minimum_steps cannot be negative")
        return value

    @field_validator(
        "fvu_relative_tolerance",
        "ev_absolute_tolerance",
        "l0_relative_tolerance",
        "improvement_rel_tol",
    )
    @classmethod
    def _positive_early_stop_tolerances(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("early-stopping tolerances must be positive")
        return value

    @field_validator("lr_reduce_factor")
    @classmethod
    def _lr_reduce_factor_valid(cls, value: float) -> float:
        if not 0.0 < value < 1.0:
            raise ValueError("lr_reduce_factor must be in (0, 1)")
        return value

    @field_validator("min_lr_scale")
    @classmethod
    def _min_lr_scale_valid(cls, value: float) -> float:
        if not 0.0 < value <= 1.0:
            raise ValueError("min_lr_scale must be in (0, 1]")
        return value

    @model_validator(mode="after")
    def _objective_reduces_lr_before_stopping(self) -> "EarlyStoppingCfg":
        if self.mode == "objective" and self.patience <= self.lr_plateau_patience:
            raise ValueError(
                "patience must exceed lr_plateau_patience so the learning rate is "
                "reduced before training stops"
            )
        return self


class StreamCfg(BaseModel):
    """Training from live model activations instead of an activation cache.

    The frozen model runs continuously on fresh documents; only a fixed
    validation set lives on disk. Each forward pass yields
    ``cache.tokens_per_fwd`` activations per site, split into whole
    ``sae.batch_size`` batches, so resuming needs only the token stream state.
    """

    enabled: bool = False
    model_device: str = "cuda:0"
    sae_device: str = "cuda:0"
    # Hard cap on streamed training tokens per site, regardless of early
    # stopping. Extra raw exports are written at each milestone token count.
    token_cap: int | None = None
    milestone_tokens: list[int] = Field(default_factory=list)
    # Stop cleanly (checkpoint + exit) after this much wall time, so a job can
    # end before a scheduler time limit and be resumed.
    stop_after_seconds: float | None = None

    @field_validator("token_cap")
    @classmethod
    def _token_cap_positive(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("stream.token_cap must be positive when set")
        return value

    @field_validator("milestone_tokens")
    @classmethod
    def _milestones_sorted_positive(cls, value: list[int]) -> list[int]:
        if any(v <= 0 for v in value):
            raise ValueError("stream.milestone_tokens must all be positive")
        if list(value) != sorted(set(value)):
            raise ValueError("stream.milestone_tokens must be strictly increasing")
        return value

    @field_validator("stop_after_seconds")
    @classmethod
    def _stop_after_seconds_nonnegative(cls, value: float | None) -> float | None:
        if value is not None and value < 0:
            raise ValueError("stream.stop_after_seconds cannot be negative")
        return value


class VLLMCfg(BaseModel):
    """Engine settings for the vLLM extraction backend.

    Chunked prefill is always disabled so every prompt is prefilled whole in
    one forward pass. Prefix caching stays off unless explicitly validated.
    """

    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.5
    # None: data.seq_len + 1 (the prompt plus the single generated token).
    max_model_len: int | None = None
    # Sequences submitted per generate() call; each becomes one vLLM request.
    # "auto" (when the captures stay on the GPU): as many as the VRAM left after
    # one probe step holds, keeping vram_headroom_gib free.
    sequences_per_call: int | Literal["auto"] = 64
    vram_headroom_gib: float = 5.0
    # Sequences per forward pass (the scheduler's token budget); None: sequences_per_call.
    max_sequences_per_step: int | None = None
    # vLLM's KV-cache pool (kv_cache_memory_bytes); None: whatever gpu_memory_utilization leaves.
    kv_cache_gib: float | None = None
    enable_prefix_caching: bool = False
    # Forward hooks do not run inside replayed CUDA graphs.
    enforce_eager: bool = True

    @field_validator("tensor_parallel_size", "sequences_per_call", "max_sequences_per_step", "kv_cache_gib")
    @classmethod
    def _positive_counts(cls, value: float | str | None) -> float | str | None:
        if isinstance(value, (int, float)) and value <= 0:
            raise ValueError("vllm counts and sizes must be positive")
        return value

    @model_validator(mode="after")
    def _auto_needs_a_step_budget(self) -> "VLLMCfg":
        if self.sequences_per_call == "auto" and self.max_sequences_per_step is None:
            raise ValueError("vllm.sequences_per_call: auto needs vllm.max_sequences_per_step")
        return self

    @field_validator("gpu_memory_utilization")
    @classmethod
    def _utilization_fraction(cls, value: float) -> float:
        if not 0.0 < value <= 1.0:
            raise ValueError("vllm.gpu_memory_utilization must be in (0, 1]")
        return value


class ActivationSinkCfg(BaseModel):
    """Where extracted activations are written and how long they live.

    ``path`` may reference environment variables (e.g.
    ``/dev/shm/sae-${SLURM_JOB_ID}``); a RAM-backed tmpfs suits streaming,
    node-local disk suits debugging. ``lifecycle``: ``consume`` deletes each
    file once read, ``cache`` keeps them.
    """

    path: str = "/dev/shm/sae-activations"
    lifecycle: SinkLifecycle = "consume"

    def resolved_path(self) -> Path:
        """The sink directory with ``~`` and environment variables expanded."""
        expanded = os.path.expanduser(os.path.expandvars(self.path))
        if "$" in expanded:
            raise ValueError(f"Unresolved environment variable in sink path {self.path!r}: {expanded!r}")
        return Path(expanded)


class RemoteCfg(BaseModel):
    """Control-store address for ``vllm_remote``: activations come from a producer process.

    The trainer hosts the store and the producer (``cli.serve_activations``,
    same config) connects to it. Both run on one node, so the activations go
    GPU to GPU; jobs sharing a node need distinct ports.
    """

    host: str = "127.0.0.1"
    port: int = 29555


class ExtractionCfg(BaseModel):
    backend: ExtractionBackend = "hf_hooks"
    vllm: VLLMCfg = Field(default_factory=VLLMCfg)
    sink: ActivationSinkCfg = Field(default_factory=ActivationSinkCfg)
    remote: RemoteCfg = Field(default_factory=RemoteCfg)


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
    early_stopping: EarlyStoppingCfg = Field(default_factory=EarlyStoppingCfg)
    stream: StreamCfg = Field(default_factory=StreamCfg)
    extraction: ExtractionCfg = Field(default_factory=ExtractionCfg)
    log: LogCfg = Field(default_factory=LogCfg)

    @model_validator(mode="after")
    def _stream_forward_yields_whole_batches(self) -> "PipelineCfg":
        if self.stream.enabled and self.cache.tokens_per_fwd % self.sae.batch_size:
            raise ValueError(
                "cache.tokens_per_fwd must be a multiple of sae.batch_size for streaming "
                f"(tokens_per_fwd={self.cache.tokens_per_fwd}, batch_size={self.sae.batch_size})"
            )
        return self

    @model_validator(mode="after")
    def _vllm_context_fits_a_sequence(self) -> "PipelineCfg":
        max_model_len = self.extraction.vllm.max_model_len
        if max_model_len is not None and max_model_len < self.data.seq_len + 1:
            raise ValueError(
                f"extraction.vllm.max_model_len={max_model_len} must be at least data.seq_len + 1 "
                f"({self.data.seq_len + 1}): each request is one sequence plus one generated token"
            )
        return self

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
