from sae_pipeline.cli.bench_sae_training import bench_sae_training
from sae_pipeline.config import PipelineCfg


def test_reports_throughput_for_every_site_on_cpu():
    cfg = PipelineCfg.from_yaml("configs/dev.yaml").with_overrides(
        **{"sae.d_sae": 32, "sae.l0_target": 4, "sae.batch_size": 16, "sae.lr_schedule": "warmup_constant"}
    )

    report = bench_sae_training(cfg, sites=2, d_in=8, steps=3, warmup=1, device="cpu")

    assert report["sites"] == 2 and report["steps"] == 3
    assert report["tokens_per_s_per_site"] > 0
    assert report["tokens_per_s_per_site"] == report["steps_per_s"] * 16
