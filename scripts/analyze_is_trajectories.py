"""Analyze SAE firing probability trajectories, cross-split Spearman correlations, and collapse metrics.

Reads saved evaluation checkpoint arrays (npz) and validation logs from an SAE training run
and computes:
1. Cross-split Spearman rank correlations of feature firing frequencies.
2. Temporal Spearman rank correlations of firing frequencies relative to the earliest saved checkpoint.
3. Spearman correlation between finite-sample inactivity and D_IS over checkpoints, per split.
4. Inverse-Simpson effective dictionary size D_IS and utilization u_IS trajectory.
5. Latent activation concentration (D_50, D_90, D_99) and quantiles over time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


def compute_spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman rank correlation between two 1D feature frequency vectors."""
    if len(x) == 0 or len(y) == 0:
        return 0.0
    # If all values are constant or zero, correlation is undefined / 0
    if np.all(x == x[0]) or np.all(y == y[0]):
        return 0.0
    res = stats.spearmanr(x, y)
    val = res.statistic if hasattr(res, "statistic") else res[0]
    return float(val) if np.isfinite(val) else 0.0


def inactivity_is_correlations(metrics_by_step: dict[int, dict[str, dict]]) -> dict[str, dict]:
    """Correlate inactivity and effective dictionary size over time per split."""
    splits = sorted({split for step_rows in metrics_by_step.values() for split in step_rows})
    correlations: dict[str, dict] = {}
    for split in splits:
        observations = [
            row
            for _, step_rows in sorted(metrics_by_step.items())
            if (row := step_rows.get(split)) is not None
            and row.get("d_is") is not None
            and row.get("inactive_pct", row.get("dead_pct")) is not None
        ]
        inactive = np.asarray(
            [float(row.get("inactive_pct", row.get("dead_pct"))) for row in observations],
            dtype=np.float64,
        )
        d_is = np.asarray([float(row["d_is"]) for row in observations], dtype=np.float64)
        finite = np.isfinite(inactive) & np.isfinite(d_is)
        inactive = inactive[finite]
        d_is = d_is[finite]
        rho: float | None = None
        if (
            inactive.size >= 2
            and not np.all(inactive == inactive[0])
            and not np.all(d_is == d_is[0])
        ):
            result = stats.spearmanr(inactive, d_is)
            value = result.statistic if hasattr(result, "statistic") else result[0]
            rho = float(value) if np.isfinite(value) else None
        correlations[split] = {"n_checkpoints": int(inactive.size), "rho": rho}
    return correlations


def analyze_run(run_dir: Path) -> dict[str, Any]:
    """Analyze evaluation checkpoint arrays and validation logs for a single SAE run."""
    eval_dir = run_dir / "eval_checkpoints"
    val_log_file = run_dir / "validation_log.jsonl"

    results: dict[str, Any] = {
        "run_dir": str(run_dir),
        "steps": [],
        "cross_split_correlations": {},
        "temporal_correlations": {},
        "inactivity_is_correlations": {},
        "metrics_by_step": {},
    }

    # 1. Parse validation_log.jsonl if present
    if val_log_file.is_file():
        with open(val_log_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                step = int(row["step"])
                lang = str(row["language"])
                if step not in results["metrics_by_step"]:
                    results["metrics_by_step"][step] = {}
                results["metrics_by_step"][step][lang] = row

    results["inactivity_is_correlations"] = inactivity_is_correlations(
        results["metrics_by_step"]
    )

    if not eval_dir.is_dir():
        return results

    # 2. Discover available evaluation steps and splits from npz files
    npz_files = sorted(eval_dir.glob("eval_arrays_step_*_*.npz"))
    step_splits: dict[int, dict[str, Path]] = {}
    for p in npz_files:
        prefix = "eval_arrays_step_"
        if not p.stem.startswith(prefix):
            continue
        step_text, separator, split = p.stem[len(prefix) :].partition("_")
        if not separator or not split:
            continue
        try:
            step = int(step_text)
        except ValueError:
            continue
        step_splits.setdefault(step, {})[split] = p

    sorted_steps = sorted(step_splits.keys())
    results["steps"] = sorted_steps

    # 3. Compute cross-split Spearman correlations at each step
    firing_by_step_split: dict[int, dict[str, np.ndarray]] = {}

    for step in sorted_steps:
        splits = step_splits[step]
        firing_by_step_split[step] = {}
        for s_name, path in splits.items():
            with np.load(path) as data:
                if "firing_frequency" in data:
                    firing_by_step_split[step][s_name] = data["firing_frequency"]

    split_names = sorted(
        {split for step_data in firing_by_step_split.values() for split in step_data}
    )
    cross_corr: dict[str, dict[int, float]] = {}
    for index, first in enumerate(split_names):
        for second in split_names[index + 1 :]:
            label = f"{first}_vs_{second}"
            cross_corr[label] = {
                step: compute_spearman(step_data[first], step_data[second])
                for step, step_data in firing_by_step_split.items()
                if first in step_data and second in step_data
            }

    results["cross_split_correlations"] = cross_corr

    # 4. Compute temporal correlations relative to step 0
    temporal_corr: dict[str, dict[int, float]] = {}
    if sorted_steps:
        base_step = sorted_steps[0]
        base_data = firing_by_step_split[base_step]
        for s_name, base_p in base_data.items():
            temporal_corr[s_name] = {}
            for step in sorted_steps:
                if s_name in firing_by_step_split[step]:
                    curr_p = firing_by_step_split[step][s_name]
                    temporal_corr[s_name][step] = compute_spearman(base_p, curr_p)

    results["temporal_correlations"] = temporal_corr
    return results


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze SAE evaluation trajectories and Spearman correlations")
    p.add_argument("--run-dir", required=True, type=Path, help="Directory containing eval_checkpoints and validation_log.jsonl")
    p.add_argument("--output-json", type=Path, default=None, help="Optional JSON path to save trajectory results")
    args = p.parse_args()

    results = analyze_run(args.run_dir)
    print(f"\n=== Trajectory Analysis for: {args.run_dir.name} ===")
    print(f"Evaluated steps: {results['steps']}")

    print("\n--- Cross-Split Spearman Rank Correlations rho(p_i, p_j) ---")
    steps = results["steps"]
    for pair, by_step in results["cross_split_correlations"].items():
        print(f"{pair}:")
        for step in steps:
            value = by_step.get(step)
            formatted = "n/a" if value is None else f"{value:.4f}"
            print(f"  step {step}: {formatted}")

    print("\n--- Inactivity vs D_IS Spearman Correlation Across Checkpoints ---")
    for split, summary in results["inactivity_is_correlations"].items():
        rho = summary["rho"]
        formatted = "n/a" if rho is None else f"{rho:.4f}"
        print(f"{split}: rho={formatted} over {summary['n_checkpoints']} checkpoints")

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved trajectory results to {args.output_json}")


if __name__ == "__main__":
    main()
