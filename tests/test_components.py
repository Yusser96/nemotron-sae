from types import SimpleNamespace

import pytest

from sae_pipeline.config import HookTargetCfg
from sae_pipeline.hooks.components import ComponentSpec, select_sites, site_requests
from sae_pipeline.model.topology import HookSite


def test_site_requests_builds_a_spec_and_site_per_slug():
    sites = [
        HookTargetCfg(slug="L5_resid_post", layer=5, component="resid_post", hook_name="layers.5.resid_post"),
        HookTargetCfg(slug="L5_expert_2", layer=5, component="expert.2", hook_name="layers.5.experts.2"),
    ]

    requests = site_requests(sites)

    assert set(requests) == {"L5_resid_post", "L5_expert_2"}
    spec, site = requests["L5_resid_post"]
    assert spec == ComponentSpec.parse(5, "resid_post")
    assert site == HookSite("resid_post", 5, "layers.5.resid_post")
    spec, site = requests["L5_expert_2"]
    assert spec == ComponentSpec.parse(5, "expert.2")
    assert site == HookSite("expert.2", 5, "layers.5.experts.2")


def test_site_requests_of_no_sites_is_empty():
    assert site_requests([]) == {}


def test_select_sites_keeps_config_order_and_rejects_unknown_slugs():
    sites = [SimpleNamespace(slug=slug) for slug in ("a", "b", "c")]

    assert [site.slug for site in select_sites(sites, ["c", "a"])] == ["a", "c"]
    assert select_sites(sites, None) == sites
    with pytest.raises(ValueError, match="not in target.sites"):
        select_sites(sites, ["d"])
