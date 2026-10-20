from types import SimpleNamespace

import pytest
import torch

from sae_pipeline.cache.manifest import CacheManifest, validation_dir_for
from sae_pipeline.cache.writer import ShardWriter
from sae_pipeline.cli.train_sae_streaming import _validation_cache_dirs
from sae_pipeline.config import PipelineCfg


def _cfg(tmp_path) -> PipelineCfg:
    return PipelineCfg.from_yaml("configs/dev.yaml").with_overrides(
        **{"cache.cache_dir": str(tmp_path), "model.name": "model", "eval.trajectory_n_tokens": 4}
    )


def _write_validation(cfg: PipelineCfg, language: str, *, layer: int = 3, component: str = "resid_post") -> None:
    path = validation_dir_for(cfg.cache.cache_dir, cfg.run_id, language, "L3_resid_post")
    manifest = CacheManifest(
        run_id=cfg.run_id,
        model="model",
        dtype="float32",
        layer=layer,
        component=component,
        d_activation=5,
        partition="validation",
        language=language,
    )
    with ShardWriter(out_dir=path, d_activation=5, shard_size_bytes=1024, dtype=torch.float32, manifest=manifest) as w:
        w.add(torch.zeros(4, 5))
    w.manifest.complete = True
    w.manifest.write(path / "manifest.json")


TARGET = SimpleNamespace(slug="L3_resid_post", layer=3, component="resid_post")


def test_validation_caches_are_found_per_language(tmp_path):
    cfg = _cfg(tmp_path)
    for language in ("de", "en"):
        _write_validation(cfg, language)

    dirs, d_in = _validation_cache_dirs(cfg, TARGET, ["de", "en"])

    assert d_in == 5
    assert dirs == {lang: validation_dir_for(tmp_path, cfg.run_id, lang, "L3_resid_post") for lang in ("de", "en")}


def test_a_validation_cache_for_another_site_is_rejected(tmp_path):
    cfg = _cfg(tmp_path)
    _write_validation(cfg, "de", layer=4)

    with pytest.raises(SystemExit, match="site=\\(4,resid_post\\)"):
        _validation_cache_dirs(cfg, TARGET, ["de"])


def test_a_missing_validation_cache_is_reported(tmp_path):
    with pytest.raises(SystemExit, match="Missing validation manifest"):
        _validation_cache_dirs(_cfg(tmp_path), TARGET, ["de"])
