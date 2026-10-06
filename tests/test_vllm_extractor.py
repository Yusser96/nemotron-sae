"""VLLMExtractor against a fake engine that writes files like vLLM's connector."""

import itertools
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from sae_pipeline.config import ActivationSinkCfg, VLLMCfg
from sae_pipeline.extraction.vllm import VLLMExtractor
from sae_pipeline.hooks.components import ComponentSpec

D = 4
SPECS = {
    "L2_resid_post": ComponentSpec.parse(2, "resid_post"),
    "L0_resid_pre": ComponentSpec.parse(0, "resid_pre"),
}


def _hidden(token: int, layer_id: int) -> torch.Tensor:
    return torch.full((D,), float(token * 100 + layer_id))


class _FakeEngine:
    def __init__(self, *, shuffle: bool = False, **engine_kwargs) -> None:
        self.kwargs = engine_kwargs
        self.shuffle = shuffle
        self.calls: list[int] = []
        self.counter = itertools.count()
        self.directory = Path(engine_kwargs["kv_transfer_config"]["kv_connector_extra_config"]["shared_storage_path"])
        self.layer_ids = engine_kwargs["speculative_config"]["draft_model_config"]["hf_config"][
            "eagle_aux_hidden_state_layer_ids"
        ]

    def generate(self, prompts, sampling_params, use_tqdm):
        assert sampling_params == "one-token"
        self.calls.append(len(prompts))
        outputs = []
        for prompt in prompts:
            tokens = prompt["prompt_token_ids"] + [0]  # plus the generated token
            hidden = torch.stack([torch.stack([_hidden(t, l) for l in self.layer_ids]) for t in tokens])
            path = self.directory / f"req-{next(self.counter)}.safetensors"
            save_file({"token_ids": torch.tensor(tokens), "hidden_states": hidden}, str(path))
            outputs.append(SimpleNamespace(kv_transfer_params={"hidden_states_path": str(path)}))
        return outputs[::-1] if self.shuffle else outputs


def _extractor(tmp_path, *, shuffle=False, sequences_per_call=2) -> VLLMExtractor:
    return VLLMExtractor(
        model_name="example/model",
        dtype="bfloat16",
        trust_remote_code=True,
        specs=SPECS,
        num_hidden_layers=4,
        seq_len=3,
        vllm_cfg=VLLMCfg(sequences_per_call=sequences_per_call),
        sink_cfg=ActivationSinkCfg(path=str(tmp_path / "sink")),
        engine_factory=lambda **kwargs: _FakeEngine(shuffle=shuffle, **kwargs),
        sampling_params_factory=lambda: "one-token",
    )


def test_engine_is_configured_for_hidden_state_extraction(tmp_path):
    kwargs = _extractor(tmp_path).llm.kwargs

    assert kwargs["enable_chunked_prefill"] is False
    assert kwargs["enable_prefix_caching"] is False
    assert kwargs["pipeline_parallel_size"] == 1
    assert kwargs["max_model_len"] == 4
    assert kwargs["speculative_config"]["method"] == "extract_hidden_states"
    assert kwargs["speculative_config"]["draft_model_config"]["hf_config"]["eagle_aux_hidden_state_layer_ids"] == [0, 3]
    assert kwargs["kv_transfer_config"]["kv_connector"] == "ExampleHiddenStatesConnector"


def test_extract_returns_sequence_major_site_activations_and_consumes_files(tmp_path):
    extractor = _extractor(tmp_path)
    token_ids = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    batch = extractor.extract(token_ids)

    assert extractor.llm.calls == [2, 1]  # sequences_per_call=2
    flat = token_ids.reshape(-1).tolist()
    torch.testing.assert_close(batch.activations["L0_resid_pre"], torch.stack([_hidden(t, 0) for t in flat]))
    torch.testing.assert_close(batch.activations["L2_resid_post"], torch.stack([_hidden(t, 3) for t in flat]))
    assert list(extractor.sink.path.iterdir()) == []
    extractor.close()
    assert not extractor.sink.path.exists()


def test_outputs_matched_to_the_wrong_prompt_are_detected(tmp_path):
    extractor = _extractor(tmp_path, shuffle=True)

    with pytest.raises(ValueError, match="token_ids"):
        extractor.extract(torch.tensor([[1, 2, 3], [4, 5, 6]]))


def test_only_residual_sites_are_supported(tmp_path):
    extractor = _extractor(tmp_path)

    assert extractor.supports(ComponentSpec.parse(1, "resid_post"))
    assert not extractor.supports(ComponentSpec.parse(1, "moe_out"))
    assert not extractor.supports(ComponentSpec.parse(9, "resid_post"))
