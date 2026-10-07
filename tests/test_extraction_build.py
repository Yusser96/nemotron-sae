from sae_pipeline.config import PipelineCfg
from sae_pipeline.extraction.build import build_extractor
from sae_pipeline.extraction.vllm import VLLMExtractor


def test_vllm_backend_gets_one_spec_per_target_site(monkeypatch):
    seen = {}

    def fake_init(self, cfg, specs, *, capture_device):
        seen.update(specs=specs, capture_device=capture_device)

    monkeypatch.setattr(VLLMExtractor, "__init__", fake_init)
    cfg = PipelineCfg.from_yaml("configs/dev.yaml").with_overrides(
        target={
            "sites": [
                {"slug": "L1_moe_out", "layer": 1, "component": "moe_out", "hook_name": "backbone.layers.1.mixer"},
                {"slug": "L2_resid_post", "layer": 2, "component": "resid_post", "hook_name": "backbone.layers.2"},
            ]
        }
    )

    extractor, tokenizer = build_extractor(cfg, "vllm", capture_device="cuda:1")

    assert isinstance(extractor, VLLMExtractor) and tokenizer is None
    assert {slug: (spec.layer, spec.kind) for slug, spec in seen["specs"].items()} == {
        "L1_moe_out": (1, "moe_out"),
        "L2_resid_post": (2, "resid_post"),
    }
    assert seen["capture_device"] == "cuda:1"
