"""Build extraction backends while keeping model ownership with the caller."""
from __future__ import annotations

from sae_pipeline.config import PipelineCfg
from sae_pipeline.extraction.base import ActivationExtractor
from sae_pipeline.hooks.components import site_requests


def build_extractor(cfg: PipelineCfg, backend: str, *, device="cuda:0", capture_device="cpu",
                    requests=None, model=None, tokenizer=None) -> tuple[ActivationExtractor, object | None]:
    requests = site_requests(cfg.target.sites or []) if requests is None else requests
    if backend in {"hf_hooks", "nemotron_extraction_hf"}:
        from sae_pipeline.model.loader import load_model_and_tokenizer

        if model is None:
            model, tokenizer = load_model_and_tokenizer(cfg.model.model_copy(update={"device_map": {"": device}}))
        model.eval()
        if backend == "hf_hooks":
            from sae_pipeline.extraction.hf import HFHookExtractor
            return HFHookExtractor(model, requests, model_device=device, capture_device=capture_device), tokenizer
        from sae_pipeline.extraction.adapters import NemotronExtractor
        return NemotronExtractor(requests, backend=backend, model=model, device=device, capture_device=capture_device), tokenizer
    if backend == "nemotron_extraction_vllm":
        from sae_pipeline.extraction.adapters import NemotronExtractor
        return NemotronExtractor(requests, backend=backend, cfg=cfg, capture_device=capture_device), None
    if backend == "vllm":
        from sae_pipeline.extraction.vllm import VLLMExtractor
        specs = {slug: spec for slug, (spec, _site) in requests.items()}
        return VLLMExtractor(cfg, specs, capture_device=capture_device), None
    raise ValueError(f"Unsupported local extraction backend: {backend}")
