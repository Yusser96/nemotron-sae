import torch

from sae_pipeline.cli.bench_extraction import bench_extraction
from sae_pipeline.extraction.base import ActivationBatch


class _CountingExtractor:
    def __init__(self) -> None:
        self.calls: list[torch.Tensor] = []

    def extract(self, token_ids: torch.Tensor) -> ActivationBatch:
        self.calls.append(token_ids)
        return ActivationBatch(token_ids, {"site": torch.zeros(token_ids.numel(), 2)})


def test_each_batch_size_gets_a_warmup_and_fresh_rows_per_timed_call():
    extractor = _CountingExtractor()
    rows = torch.arange(8 * 3).reshape(8, 3)

    results = bench_extraction(extractor, rows, [1, 2], repeats=3)

    assert [r["batch_sequences"] for r in results] == [1, 2]
    assert [r["tokens_per_call"] for r in results] == [3, 6]
    assert all(r["tokens_per_s"] > 0 for r in results)
    assert [call.shape[0] for call in extractor.calls] == [1] * 4 + [2] * 4
    assert len({tuple(call.flatten().tolist()) for call in extractor.calls[4:]}) == 4
