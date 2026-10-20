from types import SimpleNamespace

from sae_pipeline.cli.train_sae import _validation_manifest_site_mismatch
from sae_pipeline.hooks.components import ComponentSpec


def test_validation_manifest_site_matches_component_spec_kind():
    manifest = SimpleNamespace(layer=12, component="attn_out_prelinear")
    spec = ComponentSpec.parse(12, "attn_out_prelinear")

    assert _validation_manifest_site_mismatch(manifest, spec) is None


def test_validation_manifest_site_mismatch_is_reported():
    manifest = SimpleNamespace(layer=12, component="resid_post")
    spec = ComponentSpec.parse(12, "attn_out_prelinear")

    assert _validation_manifest_site_mismatch(manifest, spec) == "site=(12,resid_post)"
