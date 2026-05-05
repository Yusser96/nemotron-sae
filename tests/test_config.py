from pathlib import Path

import pytest

from sae_pipeline.config import PipelineCfg


def test_dev_config_loads():
    cfg = PipelineCfg.from_yaml("configs/dev.yaml")
    assert cfg.run_id == "dev_smoke"
    assert cfg.data.n_documents == 100
    assert cfg.data.source.startswith("nvidia/")
    assert cfg.target.layer == 25
    assert cfg.target.component == "resid_post"
    assert cfg.sae.arch == "jumprelu"
    assert cfg.sae.d_sae == 4096


def test_prod_config_loads():
    cfg = PipelineCfg.from_yaml("configs/prod.yaml")
    assert cfg.data.total_tokens is not None
    assert cfg.data.n_documents is None
    assert isinstance(cfg.target.layer, list)
    assert isinstance(cfg.sae.d_sae, list)


def test_data_cfg_requires_exactly_one_budget():
    from sae_pipeline.config import DataCfg

    with pytest.raises(ValueError):
        DataCfg(source="x", n_documents=10, total_tokens=1000)
    with pytest.raises(ValueError):
        DataCfg(source="x")


def test_overrides_overlay_dotted_fields():
    cfg = PipelineCfg.from_yaml("configs/dev.yaml")
    new = cfg.with_overrides(**{"sae.d_sae": 8192})
    assert new.sae.d_sae == 8192
    # Original untouched.
    assert cfg.sae.d_sae == 4096
