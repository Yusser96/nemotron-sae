"""Generate training and evaluation plots for SAE runs.

Two entry points:
    plot_training_curves(jsonl_path, out_dir, target_l0=None)
    plot_eval_summary(eval_json_path, arrays_npz_path, out_dir, target_l0=None)

Plots are saved as PNGs in `out_dir`. Matplotlib is forced to the headless 'Agg'
backend so this works on remote training boxes with no display.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")     # headless; safe to import on any host

import matplotlib.pyplot as plt
import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training plots
# ---------------------------------------------------------------------------


@dataclass
class _TrainSeries:
    step: np.ndarray
    loss: np.ndarray
    mse: np.ndarray
    l0_penalty: np.ndarray
    hard_l0: np.ndarray
    lr: np.ndarray
    lambda_l0: np.ndarray
    dead_pct: np.ndarray


def _read_train_log(jsonl_path: Path) -> _TrainSeries | None:
    rows: list[dict] = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        return None
    keys = ["step", "loss", "mse", "l0_penalty", "hard_l0", "lr", "lambda_l0", "dead_pct"]
    series = {k: np.array([r[k] for r in rows], dtype=np.float64) for k in keys}
    return _TrainSeries(**series)


def _safe_log_y(ax, ys: np.ndarray) -> None:
    """Use log-y when values span >1 order of magnitude and are positive."""
    pos = ys[ys > 0]
    if pos.size and pos.max() / max(pos.min(), 1e-12) > 10:
        ax.set_yscale("log")


def plot_training_curves(
    jsonl_path: str | Path,
    out_dir: str | Path,
    target_l0: int | None = None,
    title_prefix: str = "",
) -> list[Path]:
    """Read a training log and emit `training_overview.png` (multi-panel).

    Returns the list of files written.
    """
    jsonl_path = Path(jsonl_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    series = _read_train_log(jsonl_path)
    if series is None:
        log.warning("Empty training log at %s; skipping plots.", jsonl_path)
        return []

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    if title_prefix:
        fig.suptitle(title_prefix, fontsize=14)

    s = series

    # 0,0 — total loss
    ax = axes[0, 0]
    ax.plot(s.step, s.loss, color="C0")
    ax.set_title("Total loss")
    ax.set_xlabel("step")
    _safe_log_y(ax, s.loss)
    ax.grid(alpha=0.3)

    # 0,1 — MSE
    ax = axes[0, 1]
    ax.plot(s.step, s.mse, color="C1")
    ax.set_title("MSE (reconstruction)")
    ax.set_xlabel("step")
    _safe_log_y(ax, s.mse)
    ax.grid(alpha=0.3)

    # 0,2 — L0 penalty
    ax = axes[0, 2]
    ax.plot(s.step, s.l0_penalty, color="C2")
    ax.set_title("L0 penalty (unweighted)")
    ax.set_xlabel("step")
    ax.grid(alpha=0.3)

    # 0,3 — hard L0 vs target
    ax = axes[0, 3]
    ax.plot(s.step, s.hard_l0, color="C3", label="hard L0")
    if target_l0 is not None:
        ax.axhline(target_l0, color="black", linestyle="--", linewidth=1, label=f"target L0* = {target_l0}")
    ax.set_title("Sparsity (active latents per token)")
    ax.set_xlabel("step")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    # 1,0 — dead %
    ax = axes[1, 0]
    ax.plot(s.step, s.dead_pct, color="C4")
    ax.set_ylim(-2, 102)
    ax.set_title("Dead features (%)")
    ax.set_xlabel("step")
    ax.grid(alpha=0.3)

    # 1,1 — LR schedule
    ax = axes[1, 1]
    ax.plot(s.step, s.lr, color="C5")
    ax.set_title("Learning rate schedule")
    ax.set_xlabel("step")
    ax.grid(alpha=0.3)

    # 1,2 — λ schedule
    ax = axes[1, 2]
    ax.plot(s.step, s.lambda_l0, color="C6")
    ax.set_title("Sparsity coefficient λ")
    ax.set_xlabel("step")
    ax.grid(alpha=0.3)

    # 1,3 — MSE vs L0 trajectory (color = step)
    ax = axes[1, 3]
    sc = ax.scatter(s.hard_l0, s.mse, c=s.step, cmap="viridis", s=12)
    ax.set_xlabel("hard L0")
    ax.set_ylabel("MSE")
    _safe_log_y(ax, s.mse)
    ax.set_title("Trajectory: MSE vs L0 (colored by step)")
    ax.grid(alpha=0.3)
    plt.colorbar(sc, ax=ax, label="step")

    fig.tight_layout(rect=(0, 0, 1, 0.97 if title_prefix else 1.0))

    overview_path = out_dir / "training_overview.png"
    fig.savefig(overview_path, dpi=110)
    plt.close(fig)
    log.info("Wrote %s", overview_path)
    return [overview_path]


# ---------------------------------------------------------------------------
# Eval plots
# ---------------------------------------------------------------------------


def _firing_frequency_histogram(ax, freqs: np.ndarray) -> None:
    """Gemma Scope 2 Fig. 2 style: log-spaced bins, log-y count."""
    nz = freqs[freqs > 0]
    if nz.size == 0:
        ax.text(0.5, 0.5, "all latents dead", transform=ax.transAxes,
                ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        return
    lo = max(nz.min(), 1e-8)
    hi = min(nz.max(), 1.0)
    if hi <= lo:
        hi = lo * 10
    bins = np.logspace(np.log10(lo), np.log10(hi), 50)
    ax.hist(nz, bins=bins, color="C0", edgecolor="black", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("firing frequency (log)")
    ax.set_ylabel("count of latents (log)")
    ax.set_title(f"Latent firing frequency  (n={freqs.size}, dead={100.0 * (freqs == 0).mean():.1f}%)")
    ax.grid(which="both", alpha=0.3)


def _l0_histogram(ax, l0s: np.ndarray, target_l0: int | None) -> None:
    if l0s.size == 0:
        return
    bins = max(20, min(80, int(l0s.max()) + 1))
    ax.hist(l0s, bins=bins, color="C1", edgecolor="black", linewidth=0.3)
    if target_l0 is not None:
        ax.axvline(target_l0, color="black", linestyle="--", linewidth=1,
                   label=f"target L0* = {target_l0}")
        ax.legend(loc="best", fontsize=9)
    ax.set_xlabel("active latents per token")
    ax.set_ylabel("count of tokens")
    ax.set_title(f"L0 per token  (mean={l0s.mean():.1f}, std={l0s.std():.1f})")
    ax.grid(alpha=0.3)


def _recon_err_histogram(ax, errs: np.ndarray) -> None:
    if errs.size == 0:
        return
    pos = errs[errs > 0]
    if pos.size > 0 and pos.max() / max(pos.min(), 1e-12) > 100:
        bins = np.logspace(np.log10(max(pos.min(), 1e-12)), np.log10(pos.max()), 60)
        ax.set_xscale("log")
    else:
        bins = 60
    ax.hist(errs, bins=bins, color="C2", edgecolor="black", linewidth=0.3)
    ax.set_xlabel("‖x − x̂‖² per token")
    ax.set_ylabel("count of tokens")
    ax.set_title(f"Reconstruction error per token  (median={np.median(errs):.3g})")
    ax.grid(which="both", alpha=0.3)


def plot_eval_summary(
    eval_json_path: str | Path,
    arrays_npz_path: str | Path | None,
    out_dir: str | Path,
    target_l0: int | None = None,
    title_prefix: str = "",
) -> list[Path]:
    """Three histograms: firing freq, L0/token, recon-err/token."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(eval_json_path) as f:
        summary = json.load(f)

    written: list[Path] = []
    if arrays_npz_path and Path(arrays_npz_path).exists():
        arrays = np.load(arrays_npz_path)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        title = title_prefix or (
            f"L0={summary.get('l0', float('nan')):.1f}  "
            f"FVU={summary.get('fvu', float('nan')):.3f}  "
            f"dead={summary.get('dead_pct', float('nan')):.1f}%"
        )
        fig.suptitle(title, fontsize=13)
        _firing_frequency_histogram(axes[0], arrays["firing_frequency"])
        _l0_histogram(axes[1], arrays["l0_per_token"], target_l0)
        _recon_err_histogram(axes[2], arrays["recon_err_per_token"])
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        path = out_dir / "eval_histograms.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        log.info("Wrote %s", path)
        written.append(path)
    else:
        log.warning("No eval arrays at %s; only summary plot will be skipped.", arrays_npz_path)
    return written
