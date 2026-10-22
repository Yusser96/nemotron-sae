import sys

import pytest
import torch
import yaml
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from sae_pipeline.cli import dump_activations
from sae_pipeline.cli.dump_activations import sample_validation_token_ids
from sae_pipeline.config import DataSourceCfg, PipelineCfg
from sae_pipeline.data import streaming
from sae_pipeline.extraction.base import ActivationBatch


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


class _Extractor:
    closed = False

    def extract(self, token_ids):
        return ActivationBatch(token_ids, {"L2_resid_post": token_ids.flatten().float()[:, None].repeat(1, 3)})

    def close(self):
        self.closed = True


def test_adapter_backend_dumps_activations_for_existing_tokens(monkeypatch, tmp_path):
    config = tmp_path / "cfg.yml"
    cfg = _cfg(1).with_overrides(
        target={"sites": [{"slug": "L2_resid_post", "layer": 2, "component": "resid_post", "hook_name": "backbone.layers.2"}]}
    )
    config.write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
    token_path, out_path = tmp_path / "tokens.safetensors", tmp_path / "out.safetensors"
    token_ids = torch.arange(6).reshape(3, 2)
    save_file({"token_ids": token_ids}, str(token_path))
    extractor, seen = _Extractor(), {}

    def fake_build(cfg, backend, *, device):
        seen.update(backend=backend, device=device)
        return extractor, None

    monkeypatch.setattr(dump_activations, "build_extractor", fake_build)
    monkeypatch.setattr(sys, "argv", ["dump", "--config", str(config), "--backend", "nemotron_extraction_hf",
                                      "--token-ids", str(token_path), "--out", str(out_path),
                                      "--device", "cuda:3", "--batch-sequences", "2"])

    dump_activations.main()

    out = load_file(str(out_path))
    assert seen == {"backend": "nemotron_extraction_hf", "device": "cuda:3"} and extractor.closed
    assert torch.equal(out["token_ids"], token_ids)
    assert torch.equal(out["L2_resid_post"], torch.arange(6.0)[:, None].repeat(1, 3))
    with safe_open(str(out_path), "pt") as handle:
        assert handle.metadata()["backend"] == "nemotron_extraction_hf"


def test_sequence_count_must_split_evenly_over_sources(monkeypatch):
    _patch(monkeypatch)

    with pytest.raises(ValueError, match="multiple"):
        sample_validation_token_ids(_cfg(), _Tokenizer(), n_sequences=3)
