"""Opt-in adapters over the maintained activation extraction package."""

from __future__ import annotations

from sae_pipeline.extraction.base import ActivationBatch


class NemotronExtractor:
    def __init__(self, requests, *, backend, model=None, cfg=None, device="cuda:0", capture_device="cpu"):
        from nemotron_extraction import HFExtractor, HookSite, VLLMExtractor
        from nemotron_extraction.types import KINDS

        self.capture_device = capture_device
        self.model = model
        self.mapping = {}
        sites = []
        for slug, (spec, hook) in requests.items():
            if spec.kind not in KINDS:
                raise ValueError(f"Maintained extraction does not support {spec.kind}")
            site = HookSite(spec.layer, spec.kind)
            sites.append(site)
            self.mapping[slug] = site.key
        sites = tuple(dict.fromkeys(sites))
        if backend == "nemotron_extraction_hf":
            from nemotron_extraction.hooks import resolve
            from sae_pipeline.hooks.attention import output_projection

            for spec, hook in requests.values():
                path, module = resolve(model, HookSite(spec.layer, spec.kind))
                matches = hook.module_path == path
                if spec.kind == "attn_out_prelinear":
                    matches = output_projection(model.get_submodule(hook.module_path)) is module
                if not matches and not (
                    spec.kind.startswith("resid_") and hook.module_path.startswith("<residual:")
                ):
                    raise ValueError(
                        f"Configured hook {hook.module_path} differs from maintained hook {path}"
                    )
            self.extractor = HFExtractor(model, sites, device=device)
        elif backend == "nemotron_extraction_vllm":
            from sae_pipeline.sae.checkpoint_source import normalize_hf_source

            batch = cfg.extraction.vllm.sequences_per_call
            if batch == "auto":
                raise ValueError("Set numeric sequences_per_call for the opt-in maintained vLLM adapter")
            sink = cfg.extraction.sink.resolved_path()
            sink.mkdir(parents=True, exist_ok=True)
            self.extractor = VLLMExtractor(
                normalize_hf_source(cfg.model.name),
                sites,
                dtype=cfg.model.dtype,
                trust_remote_code=cfg.model.trust_remote_code,
                max_model_len=cfg.extraction.vllm.max_model_len or cfg.data.seq_len + 1,
                sequences_per_call=batch,
                sink_dir=str(sink),
                gpu_memory_utilization=cfg.extraction.vllm.gpu_memory_utilization,
                kv_cache_memory_bytes=(
                    int(cfg.extraction.vllm.kv_cache_gib * 1024**3)
                    if cfg.extraction.vllm.kv_cache_gib is not None
                    else None
                ),
            )
        else:
            raise ValueError(f"Unknown maintained extraction backend: {backend}")

    def extract(self, token_ids):
        batch = self.extractor.extract(token_ids)
        return ActivationBatch(
            batch.token_ids,
            {
                slug: batch.activations[key].to(self.capture_device, copy=True)
                for slug, key in self.mapping.items()
            },
        )

    def close(self):
        self.extractor.close()
