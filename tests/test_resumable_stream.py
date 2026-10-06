"""ResumableTokenStream: no repeats across a save/resume boundary."""

import pytest
import torch

from sae_pipeline.config import DataCfg, DataSourceCfg
from sae_pipeline.data import streaming


class _FakeIterableDataset:
    """Minimal stand-in for datasets.IterableDataset's resumable iteration."""

    def __init__(self, examples):
        self.examples = examples
        self._position = 0

    def shuffle(self, **_kwargs):
        return self

    def __iter__(self):
        # Advance _position *before* yielding: a real datasets.IterableDataset
        # reflects "N items handed out" in its state_dict even while a
        # consumer is suspended holding the N-th item, not only once it asks
        # for the (N+1)-th. Incrementing after yield (the more obvious
        # generator pattern) would under-report by one at any such
        # suspension point.
        while self._position < len(self.examples):
            item = self.examples[self._position]
            self._position += 1
            yield item

    def state_dict(self):
        return {"position": self._position}

    def load_state_dict(self, state):
        self._position = state["position"]


class _Tokenizer:
    """Encodes "de-N" / "en-N" to distinguishable, EOS-disjoint integers.

    Each document is exactly one content token; with seq_len=2 and the
    default EOS separator, one document packs into exactly one sequence
    [content_token, eos_token] -- so counting sequences is the same as
    counting documents, which keeps these tests easy to reason about.
    """

    eos_token_id = -1

    def encode(self, text, add_special_tokens=False):
        prefix, n = text.split("-")
        offset = 0 if prefix == "de" else 100_000
        return [offset + int(n)]


def _cfg():
    return DataCfg(
        sources=[
            DataSourceCfg(source="german", language="de"),
            DataSourceCfg(source="english", language="en"),
        ],
        total_tokens=8,
        seq_len=2,
        shuffle_buffer=0,
        validation_fraction=0.05,
    )


def _patch_sources(monkeypatch, n_per_source=200):
    datasets = {
        "german": _FakeIterableDataset([{"id": f"de-{n}", "text": f"de-{n}"} for n in range(n_per_source)]),
        "english": _FakeIterableDataset([{"id": f"en-{n}", "text": f"en-{n}"} for n in range(n_per_source)]),
    }
    monkeypatch.setattr(streaming, "load_dataset", lambda source, **_kwargs: datasets[source])
    monkeypatch.setattr(streaming, "_is_validation_document", lambda *_args: False)
    return datasets


def test_next_batch_round_robins_sources_without_repeats(monkeypatch):
    _patch_sources(monkeypatch)
    cfg = _cfg()
    stream = streaming.ResumableTokenStream(cfg, _Tokenizer())

    batch = stream.next_batch(batch_size=6)
    assert batch.shape == (6, 2)
    assert (batch[:, 1] == -1).all()  # every sequence ends with the EOS separator
    values = [int(row[0]) for row in batch]
    # Sources alternate de(0), en(0), de(1), en(1), ... one document each.
    assert values == [0, 100_000, 1, 100_001, 2, 100_002]
    assert len(set(values[0::2])) == len(values[0::2])  # no repeated de documents
    assert len(set(values[1::2])) == len(values[1::2])  # no repeated en documents


def test_resume_continues_without_repeating_or_skipping(monkeypatch):
    _patch_sources(monkeypatch)
    cfg = _cfg()
    stream = streaming.ResumableTokenStream(cfg, _Tokenizer())

    first = stream.next_batch(batch_size=10)
    state = stream.state_dict()

    # A fresh stream resuming from that state must continue exactly where the
    # first one left off -- not from the beginning, and not skipping ahead.
    _patch_sources(monkeypatch)  # fresh underlying datasets, position 0
    resumed = streaming.ResumableTokenStream(cfg, _Tokenizer())
    resumed.load_state_dict(state)
    second = resumed.next_batch(batch_size=10)

    # Continuing the ORIGINAL stream should match the resumed one exactly.
    third = stream.next_batch(batch_size=10)
    assert torch.equal(second, third)

    seen = {int(row[0]) for row in first} | {int(row[0]) for row in second}
    assert len(seen) == first.shape[0] + second.shape[0]  # no document seen twice


def test_exhaustion_raises_stop_iteration(monkeypatch):
    _patch_sources(monkeypatch, n_per_source=2)
    cfg = _cfg()
    stream = streaming.ResumableTokenStream(cfg, _Tokenizer())
    stream.next_batch(batch_size=4)  # exactly exhausts both 2-document sources
    with pytest.raises(StopIteration):
        stream.next_batch(batch_size=1)
