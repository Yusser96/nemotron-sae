import pytest

from sae_pipeline.cli.dump_activations import sample_validation_token_ids
from sae_pipeline.config import DataSourceCfg, PipelineCfg
from sae_pipeline.data import streaming


class _Tokenizer:
    eos_token_id = -1

    def encode(self, text, add_special_tokens=False):
        prefix, n = text.split("-")
        return [(0 if prefix == "de" else 1000) + int(n)]


def _cfg(n_sources: int = 2) -> PipelineCfg:
    sources = [DataSourceCfg(source="german", language="de"), DataSourceCfg(source="english", language="en")]
    return PipelineCfg.from_yaml("configs/dev.yaml").with_overrides(
        **{"data.sources": [s.model_dump() for s in sources[:n_sources]], "data.seq_len": 2, "data.shuffle_buffer": 0}
    )


def _patch(monkeypatch):
    documents = {
        "german": [{"text": f"de-{n}"} for n in range(10)],
        "english": [{"text": f"en-{n}"} for n in range(10)],
    }
    monkeypatch.setattr(streaming, "load_dataset", lambda source, **_kwargs: iter(documents[source]))
    # Every even document is in the validation partition.
    monkeypatch.setattr(
        streaming, "_is_validation_document", lambda _source, _ex, text, _fraction: int(text.split("-")[1]) % 2 == 0
    )


def test_samples_validation_sequences_evenly_from_every_source(monkeypatch):
    _patch(monkeypatch)

    token_ids = sample_validation_token_ids(_cfg(), _Tokenizer(), n_sequences=4)

    assert token_ids.tolist() == [[0, -1], [2, -1], [1000, -1], [1002, -1]]


def test_sequence_count_must_split_evenly_over_sources(monkeypatch):
    _patch(monkeypatch)

    with pytest.raises(ValueError, match="multiple"):
        sample_validation_token_ids(_cfg(), _Tokenizer(), n_sequences=3)
