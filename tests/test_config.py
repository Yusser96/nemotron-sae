import pytest

from sae_pipeline.config import EarlyStoppingCfg, PipelineCfg, SAECfg


def test_dev_config_loads():
    cfg = PipelineCfg.from_yaml("configs/dev.yaml")
    assert cfg.run_id == "dev_smoke"
    assert cfg.data.n_documents == 100
    assert cfg.data.source == "HuggingFaceFW/fineweb-edu"
    assert cfg.data.name == "sample-10BT"
    assert cfg.target.layer == 25
    assert cfg.target.component == "resid_post"
    assert cfg.sae.arch == "jumprelu"
    assert cfg.sae.d_sae == 4096


def test_prod_config_loads():
    cfg = PipelineCfg.from_yaml("configs/prod.yaml")
    assert cfg.data.total_tokens is not None
    assert cfg.data.n_documents is None
    assert isinstance(cfg.target.layer, list)
    assert isinstance(cfg.sae.d_sae, list)


def test_validation_run_id_defaults_to_none():
    cfg = PipelineCfg.from_yaml("configs/dev.yaml")
    assert cfg.validation_run_id is None


def test_finetuning_config_uses_released_sites_and_bilingual_budget():
    cfg = PipelineCfg.from_yaml("configs/finetuning.example.yml")
    assert cfg.data.sources is not None
    assert cfg.model.name == "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    assert [source.source for source in cfg.data.sources] == ["nvidia/Nemotron-CC-v2.1"]
    assert [source.language for source in cfg.data.sources] == ["en"]
    assert cfg.data.total_tokens == 59_998_208
    assert cfg.data.validation_tokens_per_language == 65_536
    assert cfg.cache.max_total_bytes == 5_000_000_000_000
    assert cfg.sae.lr_schedule == "cosine_warmup_constant"
    assert cfg.sae.gradient_clip_norm == 1.0
    assert cfg.target.sites is not None
    assert len(cfg.target.sites) == 14
    assert cfg.target.sites[0].hook_name == "backbone.layers.2"


def test_inverse_simpson_config_validation():
    from sae_pipeline.config import SAECfg

    # Valid configurations
    cfg = SAECfg(
        inverse_simpson_floor=0.015,
        inverse_simpson_max_scale=0.05,
        inverse_simpson_warmup_steps=50000,
        eval_interval_steps=10000,
    )
    assert cfg.inverse_simpson_floor == 0.015
    assert cfg.inverse_simpson_max_scale == 0.05
    assert cfg.inverse_simpson_warmup_steps == 50000
    assert cfg.eval_interval_steps == 10000

    # Floor must be in [0, 1]
    with pytest.raises(ValueError):
        SAECfg(inverse_simpson_floor=-0.01)
    with pytest.raises(ValueError):
        SAECfg(inverse_simpson_floor=1.05)

    # Scale must be non-negative
    with pytest.raises(ValueError):
        SAECfg(inverse_simpson_max_scale=-0.1)

    # Warmup steps must be non-negative
    with pytest.raises(ValueError):
        SAECfg(inverse_simpson_warmup_steps=-1)

    # Eval interval must be positive
    with pytest.raises(ValueError):
        SAECfg(eval_interval_steps=0)
    with pytest.raises(ValueError):
        SAECfg(eval_interval_steps=-100)


def test_data_cfg_requires_exactly_one_budget():
    from sae_pipeline.config import DataCfg

    with pytest.raises(ValueError):
        DataCfg(source="x", n_documents=10, total_tokens=1000)
    with pytest.raises(ValueError):
        DataCfg(source="x")


def test_target_sites_require_unique_slugs():
    from sae_pipeline.config import HookTargetCfg, TargetCfg

    site = HookTargetCfg(slug="same", layer=0, component="resid_post", hook_name="layers.0")
    with pytest.raises(ValueError, match="unique"):
        TargetCfg(sites=[site, site])


def test_model_device_map_accepts_a_name_or_an_explicit_mapping():
    from sae_pipeline.config import ModelCfg

    assert ModelCfg().device_map == "auto"
    assert ModelCfg(device_map={"": "cuda:0"}).device_map == {"": "cuda:0"}


def test_overrides_overlay_dotted_fields():
    cfg = PipelineCfg.from_yaml("configs/dev.yaml")
    new = cfg.with_overrides(**{"sae.d_sae": 8192})
    assert new.sae.d_sae == 8192
    # Original untouched.
    assert cfg.sae.d_sae == 4096


def test_checkpoint_defaults_and_retention_validation():
    cfg = SAECfg()
    assert cfg.checkpoint_source is None
    assert cfg.checkpoint_mode == "finetune"
    assert cfg.checkpoint_filename is None
    assert cfg.checkpoint_revision == "main"
    assert cfg.keep_last_checkpoints == 2

    with pytest.raises(ValueError, match="keep_last_checkpoints"):
        SAECfg(keep_last_checkpoints=0)
    with pytest.raises(ValueError, match="keep_last_checkpoints"):
        SAECfg(keep_last_checkpoints=-1)


def test_combined_feature_use_strategy_is_valid():
    cfg = SAECfg(feature_use_strategy="frequency_residual_reset")
    assert cfg.feature_use_strategy == "frequency_residual_reset"


def test_unsupported_sae_architectures_are_rejected_by_config():
    """Do not let schema-valid configs fail later inside build_sae."""
    for arch in ("topk", "batchtopk", "matryoshka"):
        with pytest.raises(ValueError):
            SAECfg(arch=arch)


def test_early_stopping_config_rejects_invalid_thresholds():
    with pytest.raises(ValueError, match="positive"):
        EarlyStoppingCfg(patience=0)
    with pytest.raises(ValueError, match="positive"):
        EarlyStoppingCfg(fvu_relative_tolerance=0.0)


def test_objective_early_stopping_config_validation():
    cfg = EarlyStoppingCfg(mode="objective", patience=10, lr_plateau_patience=3)
    assert (cfg.improvement_rel_tol, cfg.lr_reduce_factor, cfg.min_lr_scale) == (0.005, 0.5, 0.05)
    assert EarlyStoppingCfg().mode == "stability"

    with pytest.raises(ValueError, match="lr_plateau_patience"):
        EarlyStoppingCfg(mode="objective", patience=3, lr_plateau_patience=3)
    with pytest.raises(ValueError, match="lr_reduce_factor"):
        EarlyStoppingCfg(lr_reduce_factor=1.0)
    with pytest.raises(ValueError, match="min_lr_scale"):
        EarlyStoppingCfg(min_lr_scale=0.0)
    with pytest.raises(ValueError, match="positive"):
        EarlyStoppingCfg(improvement_rel_tol=0.0)


def test_stream_config_defaults_and_validation():
    from sae_pipeline.config import StreamCfg

    cfg = PipelineCfg.from_yaml("configs/dev.yaml")
    assert cfg.stream.enabled is False
    assert cfg.stream.model_device == cfg.stream.sae_device == "cuda:0"

    assert StreamCfg(milestone_tokens=[10, 20]).milestone_tokens == [10, 20]
    with pytest.raises(ValueError, match="increasing"):
        StreamCfg(milestone_tokens=[20, 10])
    with pytest.raises(ValueError, match="increasing"):
        StreamCfg(milestone_tokens=[10, 10])
    with pytest.raises(ValueError, match="positive"):
        StreamCfg(token_cap=0)
    with pytest.raises(ValueError, match="negative"):
        StreamCfg(stop_after_seconds=-1)


def test_extraction_config_defaults_to_hf_hooks():
    cfg = PipelineCfg.from_yaml("configs/dev.yaml")

    assert cfg.extraction.backend == "hf_hooks"
    assert cfg.extraction.vllm.enable_prefix_caching is False
    assert cfg.extraction.vllm.enforce_eager is True
    assert cfg.extraction.sink.lifecycle == "consume"
    assert (cfg.extraction.remote.host, cfg.extraction.remote.port) == ("127.0.0.1", 29555)


def test_sink_path_expands_environment_variables(monkeypatch):
    from sae_pipeline.config import ActivationSinkCfg

    monkeypatch.setenv("SLURM_JOB_ID", "1234")
    assert ActivationSinkCfg(path="/dev/shm/sae-${SLURM_JOB_ID}").resolved_path().name == "sae-1234"
    monkeypatch.delenv("SLURM_JOB_ID")
    with pytest.raises(ValueError, match="Unresolved"):
        ActivationSinkCfg(path="/dev/shm/sae-${SLURM_JOB_ID}").resolved_path()


def test_vllm_context_must_hold_a_sequence_plus_one_token():
    cfg = PipelineCfg.from_yaml("configs/dev.yaml")
    seq_len = cfg.data.seq_len

    cfg.with_overrides(**{"extraction.vllm.max_model_len": seq_len + 1})
    with pytest.raises(ValueError, match="max_model_len"):
        cfg.with_overrides(**{"extraction.vllm.max_model_len": seq_len})
    with pytest.raises(ValueError, match="gpu_memory_utilization"):
        cfg.with_overrides(**{"extraction.vllm.gpu_memory_utilization": 0.0})


def test_auto_call_size_needs_a_step_budget():
    cfg = PipelineCfg.from_yaml("configs/dev.yaml")

    auto = cfg.with_overrides(
        **{"extraction.vllm.sequences_per_call": "auto", "extraction.vllm.max_sequences_per_step": 32}
    )
    assert auto.extraction.vllm.sequences_per_call == "auto"
    assert auto.extraction.vllm.vram_headroom_gib == 5.0
    with pytest.raises(ValueError, match="max_sequences_per_step"):
        cfg.with_overrides(**{"extraction.vllm.sequences_per_call": "auto"})


def test_streaming_requires_whole_batches_per_forward():
    cfg = PipelineCfg.from_yaml("configs/dev.yaml")
    batch_size = cfg.sae.batch_size

    cfg.with_overrides(**{"stream.enabled": True, "cache.tokens_per_fwd": 2 * batch_size})
    with pytest.raises(ValueError, match="multiple of sae.batch_size"):
        cfg.with_overrides(**{"stream.enabled": True, "cache.tokens_per_fwd": batch_size + 1})
