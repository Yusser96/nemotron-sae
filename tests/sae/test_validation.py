"""Validation probes are deterministic and leave the cached shards untouched."""

from types import SimpleNamespace

import torch

from sae_pipeline.sae.validation import validation_batches


def test_validation_preprocessing_does_not_modify_cpu_shards():
    shard = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    original = shard.clone()
    reader = SimpleNamespace(manifest=SimpleNamespace(total_tokens=6), iter_shards=lambda: iter([shard]))
    def probe():
        return torch.cat(list(validation_batches(
            reader, n_tokens=6, batch_size=2, input_scale=0.5,
            input_center=torch.ones(2), device="cpu",
        )))
    torch.testing.assert_close(probe(), original * 0.5 - 1)
    torch.testing.assert_close(probe(), original * 0.5 - 1)
    assert torch.equal(shard, original)
