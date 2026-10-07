"""VLLMExtractor end to end against a fake engine running the real worker extension."""

from types import SimpleNamespace

import pytest
import torch

from sae_pipeline.config import PipelineCfg
from sae_pipeline.extraction.vllm import VLLMExtractor
from sae_pipeline.extraction.vllm_worker import ActivationHookExtension
from sae_pipeline.hooks.components import ComponentSpec
from tests.test_vllm_worker_extension import _CausalLM

SPECS = {
    "L0_mamba_out": ComponentSpec.parse(0, "mamba_out"),
    "L1_moe_out": ComponentSpec.parse(1, "moe_out"),
    "L0_resid_post": ComponentSpec.parse(0, "resid_post"),
}


class _FakeEngine(ActivationHookExtension):
    """Batches every call's prompts into one forward pass, plus one padding row."""

    def __init__(self, **engine_kwargs) -> None:
        torch.manual_seed(0)
        self.kwargs = engine_kwargs
        self.model_runner = SimpleNamespace(model=_CausalLM())
        self.calls: list[int] = []

    def collective_rpc(self, method, args=()):
        return [getattr(self, method)(*args)]

    def generate(self, prompts, sampling_params, use_tqdm):
        assert sampling_params == "one-token"
        self.calls.append(len(prompts))
        ids = [token for prompt in prompts for token in prompt["prompt_token_ids"]] + [0]
        positions = [p for prompt in prompts for p in range(len(prompt["prompt_token_ids"]))] + [0]
        with torch.no_grad():
            self.model_runner.model(torch.tensor(ids), torch.tensor(positions))
        return [object() for _ in prompts]


def _cfg(tmp_path, *, sequences_per_call=2) -> PipelineCfg:
    return PipelineCfg.from_yaml("configs/dev.yaml").with_overrides(
        **{
            "model.name": "example/model",
            "data.seq_len": 3,
            "extraction.vllm.sequences_per_call": sequences_per_call,
            "extraction.sink.path": str(tmp_path / "sink"),
        }
    )


def _extractor(tmp_path, *, sequences_per_call=2) -> VLLMExtractor:
    return VLLMExtractor(
        _cfg(tmp_path, sequences_per_call=sequences_per_call),
        SPECS,
        engine_factory=_FakeEngine,
        sampling_params_factory=lambda: "one-token",
    )


def _reference(model, token_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    """Each sequence run on its own, outside vLLM."""
    rows = {slug: [] for slug in SPECS}
    with torch.no_grad():
        for row in token_ids:
            embedded = model.model.embed(row)
            mamba = model.model.layers[0].mixer(embedded)
            rows["L0_mamba_out"].append(mamba)
            rows["L0_resid_post"].append(mamba + embedded)
            rows["L1_moe_out"].append(model.model.layers[1].mixer(mamba + embedded))
    return {slug: torch.cat(values) for slug, values in rows.items()}


def test_engine_is_configured_for_hooked_whole_prompt_batches(tmp_path):
    kwargs = _extractor(tmp_path, sequences_per_call=8).llm.kwargs

    assert kwargs["model"] == "example/model"
    assert kwargs["worker_extension_cls"] == "sae_pipeline.extraction.vllm_worker.ActivationHookExtension"
    assert kwargs["enable_chunked_prefill"] is False
    assert kwargs["enforce_eager"] is True
    assert kwargs["max_model_len"] == 4
    assert kwargs["max_num_batched_tokens"] == 8 * 4


def test_extract_returns_each_sites_rows_in_submission_order_and_consumes_files(tmp_path):
    extractor = _extractor(tmp_path)
    token_ids = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    batch = extractor.extract(token_ids)

    assert extractor.llm.calls == [2, 1]
    expected = _reference(extractor.llm.model_runner.model, token_ids)
    for slug in SPECS:
        torch.testing.assert_close(batch.activations[slug], expected[slug])
    assert list(extractor.sink.path.iterdir()) == []
    extractor.close()
    assert not extractor.sink.path.exists()


def test_sites_without_a_module_boundary_are_rejected_before_starting_vllm(tmp_path):
    with pytest.raises(ValueError, match="L3_expert_7"):
        VLLMExtractor(
            _cfg(tmp_path),
            {"L3_expert_7": ComponentSpec.parse(3, "expert.7")},
            engine_factory=lambda **_: pytest.fail("engine must not start"),
        )
