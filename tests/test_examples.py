import runpy
import sys
from pathlib import Path

import pytest


@pytest.mark.slow
def test_toy_sae_example_reaches_its_sparsity_target(tmp_path, monkeypatch, capsys):
    example = Path(__file__).resolve().parents[1] / "examples" / "toy_sae.py"
    monkeypatch.setattr(sys, "argv", [str(example), str(tmp_path)])
    runpy.run_path(str(example), run_name="__main__")
    summary = capsys.readouterr().out.splitlines()[-1]
    l0 = float(summary.split("L0 ")[1].split()[0])
    assert abs(l0 - 3) < 0.5, summary
