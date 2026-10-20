from copy import deepcopy

import torch

from sae_pipeline.config import DataCfg, DataSourceCfg
from sae_pipeline.data import exact_stream


class Dataset:
    def __init__(self, language):
        self.rows = [{"text": f"{language}-{index}-packed"} for index in range(40)]
        self.position = 0

    def __iter__(self):
        while self.position < len(self.rows):
            row = self.rows[self.position]
            self.position += 1
            yield row

    def state_dict(self):
        return {"position": self.position}

    def load_state_dict(self, state):
        self.position = state["position"]


class Tokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(letter) for letter in text]


def test_resume_preserves_shuffle_buffers_partial_documents_and_language_cursor(monkeypatch):
    monkeypatch.setattr(exact_stream, "_load_source_dataset", lambda cfg, source: Dataset(source.language))
    monkeypatch.setattr(exact_stream, "_is_validation_document", lambda *args: False)
    cfg = DataCfg(total_tokens=1000, seq_len=4, shuffle_buffer=3, seed=42,
                  sources=[DataSourceCfg(source="en", language="en"), DataSourceCfg(source="de", language="de")])
    stream = exact_stream.ExactTokenStream(cfg, Tokenizer())
    stream.next_batch(5)
    saved = deepcopy(stream.state_dict())
    assert any(saved["pending"]) and all(saved["buffers"])
    expected = stream.next_batch(12)
    restored = exact_stream.ExactTokenStream(cfg, Tokenizer())
    restored.load_state_dict(saved)
    assert torch.equal(restored.next_batch(12), expected)
    assert restored.state_dict() == stream.state_dict()
