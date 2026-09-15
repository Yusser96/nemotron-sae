"""Deterministic bilingual token mixing without contacting Hugging Face."""

import pytest
import torch
from torch import nn

from sae_pipeline.config import DataCfg, DataSourceCfg
from sae_pipeline.data import streaming


class _Dataset:
    def __init__(self, examples):
        self.examples = examples

    def shuffle(self, **_kwargs):
        return self

    def __iter__(self):
        return iter(self.examples)


class _Tokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [1] if text.startswith("de") else [2]


def test_multisource_loader_alternates_full_sequences(monkeypatch):
    def fake_load_dataset(source, **_kwargs):
        prefix = "de" if source == "german" else "en"
        return _Dataset([{"id": f"{prefix}-{n}", "text": prefix} for n in range(8)])

    monkeypatch.setattr(streaming, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(streaming, "_is_validation_document", lambda *_args: False)
    cfg = DataCfg(
        sources=[
            DataSourceCfg(source="german", language="de"),
            DataSourceCfg(source="english", language="en"),
        ],
        total_tokens=8,
        seq_len=2,
        shuffle_buffer=0,
    )

    batches = list(streaming.make_token_loader(cfg, _Tokenizer(), batch_size=1))
    assert [int(batch[0, 0]) for batch in batches] == [1, 2, 1, 2]
    assert all(batch.shape == (1, 2) for batch in batches)


def test_document_partition_is_stable():
    source = DataSourceCfg(source="example", language="de")
    example = {"uuid": "abc"}
    first = streaming._is_validation_document(source, example, "text", 0.05)
    assert streaming._is_validation_document(source, example, "text", 0.05) is first


def test_single_source_n_documents_budget_is_supported(monkeypatch):
    monkeypatch.setattr(
        streaming,
        "load_dataset",
        lambda *_args, **_kwargs: _Dataset(
            [{"id": f"doc-{n}", "text": "de"} for n in range(8)]
        ),
    )
    monkeypatch.setattr(streaming, "_is_validation_document", lambda *_args: False)
    cfg = DataCfg(source="german", n_documents=3, seq_len=2, shuffle_buffer=0)

    batches = list(streaming.make_token_loader(cfg, _Tokenizer(), batch_size=1))

    assert len(batches) == 3
    assert all(batch.shape == (1, 2) for batch in batches)


def test_multisource_n_documents_is_global_and_balanced(monkeypatch):
    def fake_load_dataset(source, **_kwargs):
        prefix = "de" if source == "german" else "en"
        return _Dataset([{"id": f"{prefix}-{n}", "text": prefix} for n in range(8)])

    monkeypatch.setattr(streaming, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(streaming, "_is_validation_document", lambda *_args: False)
    cfg = DataCfg(
        sources=[
            DataSourceCfg(source="german", language="de"),
            DataSourceCfg(source="english", language="en"),
        ],
        n_documents=5,
        seq_len=2,
        shuffle_buffer=0,
    )

    batches = list(streaming.make_token_loader(cfg, _Tokenizer(), batch_size=1))

    assert [int(batch[0, 0]) for batch in batches] == [1, 2, 1, 2, 1]


def test_token_budget_rejects_premature_source_exhaustion(monkeypatch):
    monkeypatch.setattr(
        streaming,
        "load_dataset",
        lambda *_args, **_kwargs: _Dataset(
            [{"id": f"doc-{n}", "text": "de"} for n in range(2)]
        ),
    )
    monkeypatch.setattr(streaming, "_is_validation_document", lambda *_args: False)
    cfg = DataCfg(source="german", total_tokens=100, seq_len=2, shuffle_buffer=0)

    with pytest.raises(RuntimeError, match="requested 100"):
        list(streaming.make_token_loader(cfg, _Tokenizer(), batch_size=1))
