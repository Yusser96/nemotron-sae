import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sae_pipeline.config import ModelCfg
from sae_pipeline.model import loader


def test_json_infinity_sentinels_become_float_bounds():
    decoded = loader._normalise_time_step_limit([0.0, {"__float__": "Infinity"}])
    assert decoded == (0.0, math.inf)
    assert loader._normalise_time_step_limit([{"__float__": "-Infinity"}, 1]) == (-math.inf, 1.0)
    assert loader._normalise_time_step_limit("not a limit") == "not a limit"


def test_runtime_config_repair_reaches_the_config_and_every_mixer():
    model = nn.Sequential(nn.Linear(1, 1), nn.Linear(1, 1))
    model.config = SimpleNamespace(time_step_limit=[0.0, {"__float__": "Infinity"}])
    model[1].time_step_limit = [0.0, {"__float__": "Infinity"}]
    loader._normalise_model_runtime_config(model)
    assert model.config.time_step_limit == (0.0, math.inf)
    assert model[1].time_step_limit == (0.0, math.inf)


@pytest.mark.parametrize("quant,dtype,expected", [
    ("auto", "float32", torch.float32),
    ("auto", "bfloat16", torch.bfloat16),
    ("bf16", "float32", torch.bfloat16),
])
def test_unquantized_loads_use_the_requested_dtype(quant, dtype, expected):
    kwargs = loader._resolve_quantization_kwargs(ModelCfg(load_quant=quant, dtype=dtype))
    assert kwargs == {"torch_dtype": expected}


def test_a_named_token_variable_must_be_set(monkeypatch):
    cfg = ModelCfg(token_env_var="SAE_TEST_TOKEN")
    monkeypatch.delenv("SAE_TEST_TOKEN", raising=False)
    with pytest.raises(ValueError, match="SAE_TEST_TOKEN"):
        loader._model_auth_kwargs(cfg)
    monkeypatch.setenv("SAE_TEST_TOKEN", "secret")
    assert loader._model_auth_kwargs(cfg) == {"token": "secret"}
    assert loader._model_auth_kwargs(ModelCfg()) == {}


def test_transformers5_tokenizer_falls_back_to_the_fast_class_and_gets_a_pad_token(monkeypatch):
    def unknown_backend(*args, **kwargs):
        raise ValueError("Tokenizer class TokenizersBackend does not exist or is not currently imported.")

    monkeypatch.setattr(loader.AutoTokenizer, "from_pretrained", unknown_backend)
    monkeypatch.setattr(loader.PreTrainedTokenizerFast, "from_pretrained",
                        lambda *args, **kwargs: SimpleNamespace(pad_token_id=None, eos_token_id=7))
    assert loader.load_tokenizer(ModelCfg(name="example/model")).pad_token_id == 7


def test_other_tokenizer_errors_are_not_masked(monkeypatch):
    def broken(*args, **kwargs):
        raise ValueError("something else")

    monkeypatch.setattr(loader.AutoTokenizer, "from_pretrained", broken)
    with pytest.raises(ValueError, match="something else"):
        loader.load_tokenizer(ModelCfg(name="example/model"))
