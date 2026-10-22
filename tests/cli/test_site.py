import argparse
from types import SimpleNamespace

from sae_pipeline.cli._site import cli_site, sae_variant, variant_stem
from sae_pipeline.config import PipelineCfg


def test_exact_site_preserves_custom_slug_and_hook_path():
    cfg = PipelineCfg.from_yaml("configs/finetuning.example.yml")
    target = cfg.target.sites[2]
    target.slug = "published_site_name"
    spec, hook = cli_site(cfg, SimpleNamespace(site=target.slug, layer=None, component=None), argparse.ArgumentParser())
    assert spec.slug == target.slug
    assert hook.module_path == target.hook_name


def test_sae_variant_uses_first_sweep_entry_and_cli_overrides_it():
    cfg = SimpleNamespace(sae=SimpleNamespace(arch="jumprelu", d_sae=[16384, 32768], l0_target=[50, 100]))
    assert sae_variant(cfg) == ("jumprelu", 16384, 50)
    assert sae_variant(cfg, SimpleNamespace(arch=None, width=8192, l0=20)) == ("jumprelu", 8192, 20)


def test_sae_variant_accepts_scalar_config_and_names_checkpoint_directory():
    cfg = SimpleNamespace(sae=SimpleNamespace(arch="jumprelu", d_sae=16384, l0_target=50))
    arch, width, l0 = sae_variant(cfg)
    assert (arch, width, l0) == ("jumprelu", 16384, 50)
    assert variant_stem(arch, width, l0) == "jumprelu_w16384_l0_50"
