"""Golden-value regression test for the cached ``train_sae`` step math.

Pins the per-step training log and the final weights of two short, fully
deterministic CPU runs, so refactors of the loss/optimiser code can be checked
for unchanged behaviour. One run exercises the Inverse-Simpson regulariser
with a step-count-independent LR schedule, the other the feature-use
frequency penalty with cosine decay.
"""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from sae_pipeline.cache.manifest import CacheManifest
from sae_pipeline.cache.writer import ShardWriter
from sae_pipeline.config import SAECfg
from sae_pipeline.sae.train import train_sae

D_IN, D_SAE, L0_TARGET = 8, 32, 4
LOGGED = (
    "loss",
    "reconstruction_loss",
    "l0_penalty",
    "hard_l0",
    "lr",
    "lambda_l0",
    "theta_grad_norm",
    "encoder_grad_norm",
    "ste_window_pct",
    "feature_use_penalty",
    "inverse_simpson_penalty",
    "d_is",
)

RECIPES = {
    "inverse_simpson": dict(
        lr_schedule="cosine_warmup_constant",
        reconstruction_loss="vector_sum",
        gradient_clip_norm=1.0,
        inverse_simpson_floor=1.0,
        inverse_simpson_max_scale=50.0,
        inverse_simpson_warmup_steps=3,
    ),
    "frequency_penalty": dict(
        lr_schedule="cosine_decay",
        reconstruction_loss="coordinate_mean",
        feature_use_strategy="frequency",
        feature_frequency_start_step=2,
        feature_frequency_threshold=0.05,
        feature_frequency_ema_decay=0.5,
    ),
}

# Captured from the pre-refactor implementation (commit 5b5d34f) with
# ``python -m tests.sae.test_train_golden`` -- regenerate only for an intended
# behaviour change.
GOLDEN_PATH = Path(__file__).parent / "golden" / "train_sae.json"


def _write_cache(path: Path) -> None:
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(256, D_IN, generator=generator) * 3.0
    manifest = CacheManifest(
        run_id="golden",
        model="synthetic",
        dtype="float32",
        layer=0,
        component="resid_post",
        d_activation=D_IN,
        partition="train",
    )
    with ShardWriter(
        out_dir=path, d_activation=D_IN, shard_size_bytes=2048, dtype=torch.float32, manifest=manifest
    ) as writer:
        writer.add(x)
    writer.manifest.complete = True
    writer.manifest.write(path / "manifest.json")


def run_recipe(name: str, tmp_path: Path) -> dict[str, object]:
    cache = tmp_path / "cache"
    _write_cache(cache)
    out_dir = tmp_path / name
    final = train_sae(
        cfg=SAECfg(
            arch="jumprelu",
            d_sae=D_SAE,
            l0_target=L0_TARGET,
            lr=1.0e-2,
            batch_size=16,
            n_steps=6,
            warmup_steps=2,
            l0_warmup_steps=3,
            bandwidth=0.05,
            ckpt_every=6,
            log_every=1,
            n_batches_in_buffer=4,
            seed=7,
            **RECIPES[name],
        ),
        cache_dir=cache,
        d_in=D_IN,
        arch="jumprelu",
        d_sae=D_SAE,
        l0_target=L0_TARGET,
        out_dir=out_dir,
        device="cpu",
    )
    rows = [json.loads(line) for line in (out_dir / "train_log.jsonl").read_text().splitlines()]
    weights = load_file(str(final))
    return {
        "steps": {key: [row[key] for row in rows] for key in LOGGED},
        "weights": {key: [float(value.sum()), float(value.norm())] for key, value in sorted(weights.items())},
    }


@pytest.mark.parametrize("name", sorted(RECIPES))
def test_train_sae_matches_golden_values(name, tmp_path, monkeypatch):
    monkeypatch.setattr("sae_pipeline.sae.train.plot_training_curves", lambda **_: None)
    actual = run_recipe(name, tmp_path)
    expected = json.loads(GOLDEN_PATH.read_text())[name]
    for key, values in expected["steps"].items():
        assert actual["steps"][key] == pytest.approx(values, rel=1e-4, abs=1e-6), key
    assert actual["weights"].keys() == expected["weights"].keys()
    for key, values in expected["weights"].items():
        assert actual["weights"][key] == pytest.approx(values, rel=1e-4, abs=1e-6), key


if __name__ == "__main__":
    import tempfile

    import sae_pipeline.sae.train as train_module

    train_module.plot_training_curves = lambda **_: None
    golden = {}
    for recipe in sorted(RECIPES):
        with tempfile.TemporaryDirectory() as directory:
            golden[recipe] = run_recipe(recipe, Path(directory))
    GOLDEN_PATH.parent.mkdir(exist_ok=True)
    GOLDEN_PATH.write_text(json.dumps(golden, indent=1) + "\n")
