import runpy
from pathlib import Path

import numpy as np
import pytest


correlation = runpy.run_path(str(Path(__file__).parents[1] / "scripts/analyze_is_trajectories.py"))["compute_spearman"]


def test_average_ties_preserve_spearman_correlations_without_scipy():
    assert correlation(np.array([1, 1, 2]), np.array([1, 2, 3])) == pytest.approx(3**0.5 / 2)
    assert correlation(np.array([1, 2, 3]), np.array([3, 2, 1])) == pytest.approx(-1)
    assert correlation(np.array([1, 1]), np.array([2, 3])) == 0
    assert correlation(np.array([1, np.nan]), np.array([2, 3])) == 0
