"""VLLMExtractor end to end against a fake engine running the real worker extension."""

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file

from sae_pipeline.config import PipelineCfg
from sae_pipeline.extraction.fills import FillQueue, mark_done, published_fills
from sae_pipeline.extraction.vllm import VLLMExtractor, fit_sequences, produce_fills
from sae_pipeline.extraction.vllm_worker import ActivationHookExtension
from sae_pipeline.hooks.components import ComponentSpec
from sae_pipeline.sae.checkpoint import load_trainer_state
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
    assert kwargs["kv_cache_memory_bytes"] is None


def test_the_step_budget_and_kv_pool_can_be_set_apart_from_the_call_size(tmp_path):
    cfg = _cfg(tmp_path, sequences_per_call=8).with_overrides(
        **{"extraction.vllm.max_sequences_per_step": 2, "extraction.vllm.kv_cache_gib": 1.5}
    )

    kwargs = VLLMExtractor(cfg, SPECS, engine_factory=_FakeEngine, sampling_params_factory=str).llm.kwargs

    assert kwargs["max_num_batched_tokens"] == 2 * 4
    assert kwargs["kv_cache_memory_bytes"] == 3 * 2**29


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


def test_auto_call_size_is_only_for_captures_kept_on_the_gpu(tmp_path):
    cfg = _cfg(tmp_path).with_overrides(
        **{"extraction.vllm.sequences_per_call": "auto", "extraction.vllm.max_sequences_per_step": 2}
    )

    with pytest.raises(ValueError, match="file sink"):
        VLLMExtractor(cfg, SPECS, engine_factory=lambda **_: pytest.fail("engine must not start"))


def test_the_call_size_fills_the_vram_left_after_the_step_peak_and_headroom():
    # 10 rows per sequence, each holding 6 bytes plus 2 for one site's assembled copy.
    memory = {"free": 1000, "transient": 150, "row_bytes": 6, "largest_site_row_bytes": 2}

    assert fit_sequences(memory, seq_len=10, headroom_bytes=50) == (1000 - 150 - 50) // 80
    with pytest.raises(RuntimeError, match="Not even one"):
        fit_sequences(memory, seq_len=10, headroom_bytes=800)


class _CountingStream:
    def __init__(self) -> None:
        self.next_id = 1

    def next_batch(self, batch_size: int) -> torch.Tensor:
        rows = torch.arange(self.next_id, self.next_id + 3 * batch_size).reshape(batch_size, 3)
        self.next_id += 3 * batch_size
        return rows

    def state_dict(self) -> dict[str, int]:
        return {"next_id": self.next_id}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.next_id = state["next_id"]


def _produce(tmp_path, monkeypatch, vllm_overrides=None, **kwargs):
    for name, value in [("memory_allocated", 0), ("max_memory_allocated", 0), ("memory_reserved", 0), ("max_memory_reserved", 0)]:
        monkeypatch.setattr(torch.cuda, name, lambda value=value: value)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    # Two sequences' captures fit: 3 rows of three float32 sites of width 4, ids, positions and one site's copy.
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (2 * 3 * (3 * 16 + 16 + 16), 10**9))
    cfg = _cfg(tmp_path).with_overrides(
        **{
            "extraction.vllm.sequences_per_call": "auto",
            "extraction.vllm.max_sequences_per_step": 4,
            "extraction.vllm.vram_headroom_gib": 0,
            **{f"extraction.vllm.{key}": value for key, value in (vllm_overrides or {}).items()},
        }
    )
    engines = []

    def engine(**kwargs):
        engines.append(_FakeEngine(**kwargs))
        return engines[-1]

    reports = produce_fills(
        cfg, SPECS, _CountingStream(), out_dir=tmp_path / "fills", threads=2,
        engine_factory=engine, sampling_params_factory=lambda: "one-token", **kwargs,
    )
    return reports, engines[0]


def test_each_fill_is_one_vram_sized_call_written_one_file_per_site(tmp_path, monkeypatch):
    reports, engine = _produce(tmp_path, monkeypatch, fills=2)

    assert engine.calls == [4, 2, 2]
    assert [r["sequences"] for r in reports] == [2, 2] and reports[0]["tokens"] == 6
    assert all("write_gb_per_s" in report for report in reports)
    assert sorted(p.name for p in (tmp_path / "fills").iterdir()) == ["fill_000000", "fill_000001", "fills.json"]
    fill = tmp_path / "fills" / "fill_000001"
    token_ids = load_file(fill / "token_ids.safetensors")["token_ids"]
    assert token_ids.tolist() == [[7, 8, 9], [10, 11, 12]]
    assert load_trainer_state(fill / "stream_state.pt") == {"token_stream": {"next_id": 13}}
    expected = _reference(engine.model_runner.model, token_ids)
    for slug in SPECS:
        torch.testing.assert_close(load_file(fill / f"{slug}.safetensors")["activations"], expected[slug])


def test_auto_fills_stop_at_max_sequences_per_call(tmp_path, monkeypatch):
    reports, engine = _produce(tmp_path, monkeypatch, vllm_overrides={"max_sequences_per_call": 1}, fills=2)

    assert engine.calls == [4, 1, 1]
    assert [r["sequences"] for r in reports] == [1, 1]


def test_a_rerun_continues_after_the_newest_fill_or_from_a_start_state(tmp_path, monkeypatch):
    start = tmp_path / "stream_state.pt"
    torch.save({"token_stream": {"next_id": 21}}, start)
    _produce(tmp_path, monkeypatch, fills=1, start_state=start)
    (tmp_path / "fills" / "fill_000001.partial").mkdir()

    _produce(tmp_path, monkeypatch, fills=1, start_state=start)

    fills = tmp_path / "fills"
    assert sorted(p.name for p in fills.iterdir()) == ["fill_000000", "fill_000001", "fills.json"]
    assert load_file(fills / "fill_000000" / "token_ids.safetensors")["token_ids"][0, 0] == 21
    assert load_file(fills / "fill_000001" / "token_ids.safetensors")["token_ids"][0, 0] == 27


def test_a_producer_waits_for_its_consumers_and_stops_when_they_are_done(tmp_path, monkeypatch):
    queue = FillQueue(tmp_path / "fills", ["c0"], poll_seconds=0)
    monkeypatch.setattr("time.sleep", lambda _seconds: mark_done(tmp_path / "fills", "c0"))

    # Room for one fill: the second waits, and the consumer finishes meanwhile.
    reports, _engine = _produce(tmp_path, monkeypatch, queue=queue, max_buffer_bytes=1)

    assert [report["fill"] for report in reports] == [0, 1]
    assert all("write_gb_per_s" in report for report in reports)
    assert published_fills(tmp_path / "fills") == [0, 1]
