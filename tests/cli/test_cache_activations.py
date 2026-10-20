"""Bounded cache runs must leave a resumable, incomplete manifest."""

import sys
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import save_file

from sae_pipeline.cache.manifest import CacheManifest
from sae_pipeline.cli import cache_activations


class _Block(nn.Module):
    def forward(self, hidden_states):
        return hidden_states


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.backbone = nn.Module()
        self.backbone.layers = nn.ModuleList([_Block()])

    def forward(self, tokens):
        hidden = torch.zeros((*tokens.shape, 4), device=tokens.device)
        return self.backbone.layers[0](hidden)


class _Tokenizer:
    eos_token_id = 0


def test_max_batches_writes_resumable_partial_cache(tmp_path, monkeypatch):
    config = tmp_path / "config.yml"
    config.write_text(
        f"""run_id: partial
data:
  total_tokens: 4
  seq_len: 1
cache:
  cache_dir: {tmp_path / 'caches'}
  shard_size_bytes: 16
  tokens_per_fwd: 2
target:
  sites:
    - slug: L0_resid_post
      layer: 0
      component: resid_post
      hook_name: backbone.layers.0
"""
    )
    monkeypatch.setattr(cache_activations, "load_model_and_tokenizer", lambda _cfg: (_Model(), _Tokenizer()))
    monkeypatch.setattr(
        cache_activations,
        "make_token_loader",
        lambda *_args, **_kwargs: iter([torch.ones(2, 1, dtype=torch.long)]),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cache_activations", "--config", str(config), "--all-targets",
            "--max-batches", "1",
        ],
    )

    cache_activations.main()

    manifest = CacheManifest.read(tmp_path / "caches" / "partial" / "L0_resid_post" / "manifest.json")
    assert manifest.total_tokens == 2
    assert manifest.complete is False


def test_vllm_backend_caches_without_loading_the_hf_model(tmp_path, monkeypatch):
    from sae_pipeline.extraction.base import ActivationBatch
    from sae_pipeline.extraction import vllm

    class _FakeExtractor:
        closed = False

        def extract(self, token_ids):
            return ActivationBatch(token_ids, {"L0_resid_post": torch.full((token_ids.numel(), 4), 3.0)})

        def close(self):
            _FakeExtractor.closed = True

    def _no_hf_model(_cfg):
        raise AssertionError("the vllm backend must not load the HF model")

    config = tmp_path / "config.yml"
    config.write_text(
        f"""run_id: vllm
data:
  total_tokens: 4
  seq_len: 1
cache:
  cache_dir: {tmp_path / 'caches'}
  shard_size_bytes: 16
  tokens_per_fwd: 2
extraction:
  backend: vllm
target:
  sites:
    - {{slug: L0_resid_post, layer: 0, component: resid_post, hook_name: backbone.layers.0}}
"""
    )
    monkeypatch.setattr(cache_activations, "load_model_and_tokenizer", _no_hf_model)
    monkeypatch.setattr(cache_activations, "load_tokenizer", lambda _cfg: _Tokenizer())
    monkeypatch.setattr(vllm, "VLLMExtractor", lambda cfg, specs: _FakeExtractor())
    monkeypatch.setattr(
        cache_activations,
        "make_token_loader",
        lambda *_args, **_kwargs: iter([torch.ones(2, 1, dtype=torch.long), torch.ones(2, 1, dtype=torch.long)]),
    )
    monkeypatch.setattr(sys, "argv", ["cache_activations", "--config", str(config), "--all-targets"])

    cache_activations.main()

    manifest = CacheManifest.read(tmp_path / "caches" / "vllm" / "L0_resid_post" / "manifest.json")
    assert manifest.total_tokens == 4 and manifest.complete is True
    assert _FakeExtractor.closed


def test_interrupted_multi_site_cache_is_repaired_to_common_prefix(tmp_path: Path):
    cache_root = tmp_path / "caches" / "run"
    base_dir = cache_root
    manifests = {}
    for slug in ("site_a", "site_b"):
        site_dir = base_dir / slug
        site_dir.mkdir(parents=True)
        save_file({"x": torch.zeros(2, 3)}, str(site_dir / "shard_00000.safetensors"))
        shard_paths = ["shard_00000.safetensors"]
        total_tokens = 2
        if slug == "site_a":
            save_file({"x": torch.ones(2, 3)}, str(site_dir / "shard_00001.safetensors"))
            shard_paths.append("shard_00001.safetensors")
            total_tokens = 4
        manifest = CacheManifest(
            run_id="run",
            model="model",
            dtype="bfloat16",
            layer=0,
            component=slug,
            d_activation=3,
            n_shards=len(shard_paths),
            total_tokens=total_tokens,
            shard_paths=shard_paths,
        )
        manifest.write(site_dir / "manifest.json")
        manifests[slug] = manifest

    common = cache_activations._repair_misaligned_manifests(
        cache_root=cache_root,
        base_dir=base_dir,
        manifests=manifests,
        token_multiple=2,
    )

    assert common == 2
    assert manifests["site_a"].total_tokens == 2
    assert manifests["site_a"].shard_paths == ["shard_00000.safetensors"]
    assert not (base_dir / "site_a" / "shard_00001.safetensors").exists()
    recovery_dirs = list(cache_root.parent.glob(".run.recovery-*"))
    assert len(recovery_dirs) == 1
    assert (recovery_dirs[0] / "site_a" / "shard_00001.safetensors").exists()


def test_cache_recipe_and_storage_dtype_are_validated(tmp_path):
    import argparse
    import pytest
    from sae_pipeline.config import PipelineCfg
    from sae_pipeline.hooks.components import ComponentSpec
    from sae_pipeline.model.topology import HookSite
    from sae_pipeline.cache.writer import ShardWriter
    from safetensors.torch import load_file

    cfg = PipelineCfg(run_id="identity", data={"total_tokens": 4}, target={"layer": 0, "component": "resid_post"}).with_overrides(**{
        "cache.cache_dir": str(tmp_path), "model.dtype": "float16"
    })
    args = argparse.Namespace(partition="train", language=None)
    spec = ComponentSpec.parse(0, "resid_post")
    requests = {spec.slug: (spec, HookSite("resid_post", 0, "layers.0"))}
    def reconcile(config=cfg, selected=requests):
        return cache_activations._reconcile_manifests(config, args, selected, {spec.slug: spec.component}, {spec.slug: 0})
    _, base, manifests, _ = reconcile()
    with ShardWriter(out_dir=base / spec.slug, d_activation=4, shard_size_bytes=16,
                     dtype=getattr(torch, cfg.model.dtype), manifest=manifests[spec.slug]) as writer:
        writer.add(torch.ones(2, 4))
    saved = CacheManifest.read(base / spec.slug / "manifest.json")
    assert load_file(str(base / spec.slug / saved.shard_paths[0]))["x"].dtype == torch.float16
    reconcile()
    for overrides in ({"model.dtype": "float32"}, {"extraction.backend": "vllm"}, {"model.load_quant": "nf4"}):
        with pytest.raises(RuntimeError, match="incompatible"):
            reconcile(cfg.with_overrides(**overrides))
    with pytest.raises(RuntimeError, match="extraction_fingerprint"):
        reconcile(selected={spec.slug: (spec, HookSite("resid_post", 0, "other.0"))})

