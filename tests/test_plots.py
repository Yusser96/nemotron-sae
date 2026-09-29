"""Smoke-test the plotting module on synthetic training/eval artifacts."""

import json
from pathlib import Path

import numpy as np

from sae_pipeline.eval.plots import (
    plot_eval_summary,
    plot_training_curves,
    plot_validation_curves,
)


def _write_fake_train_log(path: Path, n_steps: int = 200) -> None:
    rng = np.random.default_rng(0)
    with open(path, "w") as f:
        for s in range(1, n_steps + 1):
            mse = 1.0 / (1 + s / 20)
            l0 = 30 + 5 * np.sin(s / 10) + rng.standard_normal() * 0.5
            row = {
                "step": s,
                "loss": float(mse + 0.001 * (l0 - 30) ** 2),
                "mse": float(mse),
                "l0_penalty": float((l0 - 30) ** 2),
                "hard_l0": float(max(0.0, l0)),
                "lr": 7e-5 * min(1.0, s / 50),
                "lambda_l0": min(1.0, s / 100),
                "dead_pct": float(max(0.0, 80 - 0.4 * s)),
            }
            f.write(json.dumps(row) + "\n")


def test_plot_training_curves_writes_png(tmp_path: Path):
    log_path = tmp_path / "train_log.jsonl"
    _write_fake_train_log(log_path, n_steps=300)

    out_dir = tmp_path / "plots"
    written = plot_training_curves(log_path, out_dir, target_l0=30, title_prefix="test")
    assert written, "plot_training_curves returned no files"
    assert (out_dir / "training_overview.png").exists()
    assert (out_dir / "training_overview.png").stat().st_size > 5000


def test_plot_eval_summary_writes_png(tmp_path: Path):
    rng = np.random.default_rng(1)
    d_sae = 256
    n_tokens = 4096

    eval_json = tmp_path / "eval.json"
    summary = {
        "l0": 32.0, "fvu": 0.18, "dead_pct": 5.0, "n_tokens": n_tokens,
        "arch": "jumprelu", "d_sae": d_sae, "l0_target": 30,
    }
    with open(eval_json, "w") as f:
        json.dump(summary, f)

    arrays_npz = tmp_path / "eval_arrays.npz"
    firing = rng.beta(0.5, 5.0, size=d_sae).astype(np.float32)
    firing[:10] = 0.0  # some dead
    l0_per_token = rng.poisson(32, size=n_tokens).astype(np.int32)
    recon_err = rng.lognormal(mean=0.0, sigma=0.7, size=n_tokens).astype(np.float32)
    np.savez_compressed(arrays_npz,
                        firing_frequency=firing,
                        l0_per_token=l0_per_token,
                        recon_err_per_token=recon_err)

    out_dir = tmp_path / "plots"
    written = plot_eval_summary(eval_json, arrays_npz, out_dir, target_l0=30, title_prefix="test")
    assert written
    assert (out_dir / "eval_histograms.png").exists()
    assert (out_dir / "eval_histograms.png").stat().st_size > 5000


def test_plot_training_handles_empty_log(tmp_path: Path):
    log_path = tmp_path / "train_log.jsonl"
    log_path.write_text("")
    out_dir = tmp_path / "plots"
    written = plot_training_curves(log_path, out_dir)
    assert written == []
    # Empty input should not produce a file
    assert not (out_dir / "training_overview.png").exists() or \
           (out_dir / "training_overview.png").stat().st_size == 0


def test_plot_validation_curves_separates_languages(tmp_path: Path):
    log_path = tmp_path / "validation_log.jsonl"
    rows = []
    for step, fvu in ((10_000, 0.6), (20_000, 0.5)):
        for language in ("de", "en"):
            rows.append(
                {
                    "step": step,
                    "language": language,
                    "fvu": fvu + (0.01 if language == "de" else 0.0),
                    "l0": 10.0,
                    "inactive_pct": 80.0,
                    "target_l0": 10,
                }
            )
    log_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    out_dir = tmp_path / "plots"
    written = plot_validation_curves(log_path, out_dir, title_prefix="test")
    assert written == [out_dir / "validation_evolution.png"]
    assert written[0].stat().st_size > 5_000


def test_plot_validation_curves_with_inverse_simpson(tmp_path: Path):
    log_path = tmp_path / "validation_log.jsonl"
    rows = []
    for step, fvu in ((10_000, 0.6), (20_000, 0.5)):
        for language in ("de", "en", "train"):
            rows.append(
                {
                    "step": step,
                    "language": language,
                    "fvu": fvu,
                    "l0": 10.0,
                    "inactive_pct": 50.0,
                    "d_is": 150.0,
                    "u_is": 0.009,
                    "target_l0": 10,
                }
            )
    log_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    out_dir = tmp_path / "plots"
    written = plot_validation_curves(log_path, out_dir, title_prefix="test_is")
    assert written == [out_dir / "validation_evolution.png"]
    assert written[0].stat().st_size > 5_000
