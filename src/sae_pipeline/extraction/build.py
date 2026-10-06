"""Build the configured extraction backend for a config's ``target.sites``."""

from __future__ import annotations

from sae_pipeline.config import PipelineCfg
from sae_pipeline.extraction.base import ActivationExtractor
from sae_pipeline.hooks.components import site_requests


def build_extractor(
    cfg: PipelineCfg, backend: str, *, device: str = "cuda:0", capture_device: str = "cpu"
) -> tuple[ActivationExtractor, object | None]:
    """The extractor and, for ``hf_hooks``, the tokenizer loaded with the model.

    ``device`` holds the whole HF model; vLLM places the model itself.
    """
    requests = site_requests(cfg.target.sites or [])
    if backend == "hf_hooks":
        from sae_pipeline.extraction.hf import HFHookExtractor
        from sae_pipeline.model.loader import load_model_and_tokenizer

        model, tokenizer = load_model_and_tokenizer(cfg.model.model_copy(update={"device_map": {"": device}}))
        model.eval()
        return HFHookExtractor(model, requests, model_device=device, capture_device=capture_device), tokenizer
    from sae_pipeline.extraction.vllm import VLLMExtractor

    specs = {slug: spec for slug, (spec, _site) in requests.items()}
    return VLLMExtractor.from_config(cfg, specs, capture_device=capture_device), None
