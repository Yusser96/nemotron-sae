import argparse
from types import SimpleNamespace

from sae_pipeline.cli._site import cli_site
from sae_pipeline.config import PipelineCfg


def test_exact_site_preserves_custom_slug_and_hook_path():
    cfg = PipelineCfg.from_yaml("configs/finetuning.example.yml")
    target = cfg.target.sites[2]
    target.slug = "published_site_name"
    spec, hook = cli_site(cfg, SimpleNamespace(site=target.slug, layer=None, component=None), argparse.ArgumentParser())
    assert spec.slug == target.slug
    assert hook.module_path == target.hook_name

