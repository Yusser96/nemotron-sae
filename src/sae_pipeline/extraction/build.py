"""Build the configured extraction backend for a config's ``target.sites``."""

from __future__ import annotations

from sae_pipeline.config import PipelineCfg
from sae_pipeline.extraction.base import ActivationExtractor
from sae_pipeline.hooks.components import site_requests

HF_BACKENDS = {"hf_hooks", "nemotron_extraction_hf"}
LOCAL_BACKENDS = ("hf_hooks", "vllm", "nemotron_extraction_hf", "nemotron_extraction_vllm")


def build_extractor(
    cfg: PipelineCfg,
    backend: str,
    *,
    device: str = "cuda:0",
    capture_device: str = "cpu",
    requests: dict | None = None,
    model=None,
    tokenizer=None,
) -> tuple[ActivationExtractor, object | None]:
    """The extractor and, for HF backends, the tokenizer.

    ``requests`` defaults to every site in ``target.sites``. An HF backend loads
    the model onto ``device`` unless the caller passes one; vLLM places the
    model itself.
    """
    requests = site_requests(cfg.target.sites or []) if requests is None else requests
    if backend in HF_BACKENDS:
        from sae_pipeline.model.loader import load_model_and_tokenizer

        if model is None:
            model, tokenizer = load_model_and_tokenizer(cfg.model.model_copy(update={"device_map": {"": device}}))
        model.eval()
        if backend == "hf_hooks":
            from sae_pipeline.extraction.hf import HFHookExtractor

            return HFHookExtractor(model, requests, model_device=device, capture_device=capture_device), tokenizer
        from sae_pipeline.extraction.adapters import NemotronExtractor

        extractor = NemotronExtractor(
            requests, backend=backend, model=model, device=device, capture_device=capture_device
        )
        return extractor, tokenizer
    if backend == "nemotron_extraction_vllm":
        from sae_pipeline.extraction.adapters import NemotronExtractor

        return NemotronExtractor(requests, backend=backend, cfg=cfg, capture_device=capture_device), None
    if backend == "vllm":
        from sae_pipeline.extraction.vllm import VLLMExtractor

        specs = {slug: spec for slug, (spec, _site) in requests.items()}
        return VLLMExtractor(cfg, specs, capture_device=capture_device), None
    raise ValueError(f"Unsupported local extraction backend: {backend}")
