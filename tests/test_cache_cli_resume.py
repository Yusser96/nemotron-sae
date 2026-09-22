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
