"""Cached training stops once bilingual validation metrics are stable."""

import json

import torch
from safetensors.torch import save_file

from sae_pipeline.config import EarlyStoppingCfg, SAECfg
from sae_pipeline.sae.jumprelu import JumpReLUSAE
from sae_pipeline.sae.train import train_sae
from tests.synthetic import write_cache


def test_train_stops_only_after_two_stable_post_warmup_bilingual_evals(tmp_path):
    torch.manual_seed(4)
    d_in, d_sae = 4, 8
    x = torch.randn(64, d_in)
    train_cache = tmp_path / "train"
    write_cache(train_cache, x, run_id="train", shard_size_bytes=256)

    validation_dirs = {}
    for language in ("de", "en"):
        validation_dirs[language] = tmp_path / "validation" / language
        write_cache(validation_dirs[language], x[:16], language=language, run_id="validation", shard_size_bytes=256)

    initial = JumpReLUSAE(d_in=d_in, d_sae=d_sae)
    with torch.no_grad():
        initial.W_enc.zero_()
        initial.b_enc.zero_()
        initial.b_dec.zero_()
    checkpoint = tmp_path / "initial.safetensors"
    save_file(
        {key: value.contiguous() for key, value in initial.state_dict().items()},
        str(checkpoint),
    )

    out_dir = tmp_path / "out"
    train_sae(
        cfg=SAECfg(
            arch="jumprelu",
            d_sae=d_sae,
            l0_target=0,
            lr=0.0,
            batch_size=8,
            n_steps=10,
            eval_interval_steps=1,
            warmup_steps=1,
            l0_warmup_steps=1,
            ckpt_every=5,
            log_every=1,
            n_batches_in_buffer=2,
            checkpoint_source=checkpoint,
            checkpoint_mode="finetune",
        ),
        cache_dir=train_cache,
        d_in=d_in,
        arch="jumprelu",
        d_sae=d_sae,
        l0_target=0,
        out_dir=out_dir,
        device="cpu",
        validation_cache_dirs=validation_dirs,
        validation_tokens=16,
        validation_dead_tokens=16,
        early_stopping=EarlyStoppingCfg(
            enabled=True,
            evaluation_interval_steps=1,
            minimum_steps=2,
            patience=2,
            fvu_relative_tolerance=0.01,
            ev_absolute_tolerance=0.005,
            l0_relative_tolerance=0.10,
        ),
    )

    train_rows = [json.loads(line) for line in (out_dir / "train_log.jsonl").read_text().splitlines()]
    validation_rows = [
        json.loads(line) for line in (out_dir / "validation_log.jsonl").read_text().splitlines()
    ]
    assert train_rows[-1]["step"] == 3
    assert {row["language"] for row in validation_rows} == {"train_probe", "de", "en"}
    assert all(row["probe_sample_tokens"] == 16 for row in validation_rows)
    assert all(row["inactive_sample_tokens"] == 16 for row in validation_rows)
    assert all(row["early_stop_triggered"] for row in validation_rows[-3:])
    assert (out_dir / "sae_step_0000003.safetensors").is_file()
    assert (out_dir / "plots" / "validation_evolution.png").is_file()
