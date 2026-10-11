import argparse
from types import SimpleNamespace

import pytest

from sae_pipeline.cli._site import cli_site
from sae_pipeline.cli.train_sae import _validation_manifest_site_mismatch
from sae_pipeline.config import PipelineCfg
from sae_pipeline.hooks.components import ComponentSpec


def test_exact_site_preserves_custom_slug_and_hook_path():
    cfg = PipelineCfg.from_yaml("configs/finetuning.example.yml")
    target = cfg.target.sites[2]
    target.slug = "published_site_name"
    spec, hook = cli_site(cfg, SimpleNamespace(site=target.slug, layer=None, component=None), argparse.ArgumentParser())
    assert spec.slug == target.slug
    assert hook.module_path == target.hook_name


def test_indexed_validation_site_keeps_index():
    spec = ComponentSpec.parse(3, "expert.2")
    assert _validation_manifest_site_mismatch(SimpleNamespace(layer=3, component="expert.2"), spec) is None
    assert _validation_manifest_site_mismatch(SimpleNamespace(layer=3, component="expert.1"), spec)
