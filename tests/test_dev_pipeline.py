"""End-to-end pipeline smoke test using SYNTHETIC activations (no LM, no CUDA).

Writes a fake activation cache, runs a tiny SAE training, then evaluates. Validates
that writer → reader → SAE → eval chain works end-to-end on this machine.
The full real-model run (cache_activations CLI loading the 30B Nemotron) is a separate
test that requires a CUDA host; that one is gated behind `-m cuda`.
"""

import json
from pathlib import Path

import pytest
import torch

from sae_pipeline.cache.manifest import CacheManifest
from sae_pipeline.cache.writer import ShardWriter
from sae_pipeline.config import SAECfg
from sae_pipeline.eval.metrics import reconstruction_metrics
from sae_pipeline.sae.train import build_sae, train_sae


@pytest.mark.slow
def test_synthetic_pipeline(tmp_path: Path):
    d = 64
    n_tokens = 8192
    torch.manual_seed(0)
    base = torch.randn(d, d)
    # Generate sparse-coded random activations: x = A @ z with z sparse.
    z = torch.randn(n_tokens, d)
    z = z * (z.abs() > 0.5)
    x = z @ base.T

    cache_dir = tmp_path / "cache"
    manifest = CacheManifest(
        run_id="t", model="synthetic", dtype="float32",
        layer=0, component="resid_post", d_activation=d,
    )
    with ShardWriter(
        out_dir=cache_dir, d_activation=d,
        shard_size_bytes=128 * 1024, dtype=torch.float32,
        manifest=manifest,
    ) as w:
        w.add(x)

    sae_cfg = SAECfg(
        arch="jumprelu",
        d_sae=128,
        l0_target=20,
        lr=5e-3,
        batch_size=256,
        n_steps=200,
        warmup_steps=20,
        l0_warmup_steps=50,
        ckpt_every=200,
        log_every=50,
        n_batches_in_buffer=2,
        ckpt_dir=tmp_path / "ckpts",
    )

    out_dir = tmp_path / "ckpts"
    train_sae(
        cfg=sae_cfg, cache_dir=cache_dir,
        d_in=d, arch="jumprelu", d_sae=128, l0_target=20,
        out_dir=out_dir, device="cpu",
    )

    # Verify a checkpoint exists
    ckpts = list(out_dir.glob("sae_step_*.safetensors"))
    assert ckpts, ckpts
    log_path = out_dir / "train_log.jsonl"
    assert log_path.exists()
    lines = log_path.read_text().splitlines()
    assert len(lines) > 0
    last = json.loads(lines[-1])
    # loss decreased over training
    first = json.loads(lines[0])
    assert last["mse"] < first["mse"], (first["mse"], last["mse"])

    # Reload and eval
    from safetensors.torch import load_file
    sae = build_sae("jumprelu", d_in=d, d_sae=128)
    sae.load_state_dict(load_file(str(ckpts[-1])))
    metrics = reconstruction_metrics(sae, x, dead_threshold_tokens=2048)
    assert metrics.fvu < 1.0
    assert metrics.l0 > 0
