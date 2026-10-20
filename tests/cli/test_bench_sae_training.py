import pytest
import torch
from safetensors.torch import save_file

from sae_pipeline.cli.bench_sae_training import bench_fills, bench_sae_training
from sae_pipeline.config import PipelineCfg


def _cfg() -> PipelineCfg:
    return PipelineCfg.from_yaml("configs/dev.yaml").with_overrides(
        **{"sae.d_sae": 32, "sae.l0_target": 4, "sae.batch_size": 16, "sae.lr_schedule": "warmup_constant"}
    )


def test_reports_throughput_for_every_site_on_cpu():
    cfg = _cfg()

    report = bench_sae_training(cfg, sites=2, d_in=8, steps=3, warmup=1, device="cpu")

    assert report["sites"] == 2 and report["steps"] == 3
    assert report["tokens_per_s_per_site"] > 0
    assert report["tokens_per_s_per_site"] == report["steps_per_s"] * 16


def test_fills_are_trained_site_by_site_in_every_mode(tmp_path):
    for index in range(2):
        fill = tmp_path / f"fill_{index:06d}"
        fill.mkdir()
        # 40 rows: two whole batches of 16 per site, the rest dropped.
        save_file({"activations": 50 * torch.randn(40, 8).bfloat16()}, str(fill / "a.safetensors"))
        save_file({"activations": 50 * torch.randn(40, 6).bfloat16()}, str(fill / "b.safetensors"))

    reports = bench_fills(_cfg(), sorted(tmp_path.iterdir()), ["a", "b"], modes=["stream", "memory"], device="cpu")

    assert [r["mode"] for r in reports] == ["stream", "memory"]
    assert all(r["site_steps"] == 2 * 2 * 2 and r["sites"] == 2 for r in reports)
    assert reports[0]["load_gb_per_s"] is None and reports[1]["load_gb_per_s"] > 0
    assert reports[1]["tokens_per_s"] == pytest.approx(reports[1]["steps_per_s"] * 16)
