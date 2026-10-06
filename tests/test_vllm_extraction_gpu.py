"""Real-engine check of the vLLM extraction backend (needs vLLM, a GPU and a model).

Run with ``SAE_TEST_MODEL=<hf id or path> pytest -m cuda tests/test_vllm_extraction_gpu.py -s``.
``SAE_TEST_SINK`` optionally points the sink somewhere specific (e.g. /dev/shm).
"""

import os
import shutil
from types import SimpleNamespace

import pytest
import torch

from sae_pipeline.config import ActivationSinkCfg, VLLMCfg
from sae_pipeline.hooks.components import ComponentSpec

pytestmark = [pytest.mark.cuda, pytest.mark.slow]

MODEL = os.environ.get("SAE_TEST_MODEL")
SEQ_LEN, N_SEQUENCES = 64, 2


@pytest.fixture(scope="module")
def vllm_run(tmp_path_factory):
    vllm = pytest.importorskip("vllm")
    if MODEL is None or not torch.cuda.is_available():
        pytest.skip("needs SAE_TEST_MODEL and a CUDA device")
    from safetensors.torch import load_file
    from transformers import AutoConfig

    from sae_pipeline.extraction.vllm import VLLMExtractor

    n_layers = AutoConfig.from_pretrained(MODEL, trust_remote_code=True).num_hidden_layers
    specs = {
        "L0_resid_pre": ComponentSpec.parse(0, "resid_pre"),
        "L0_resid_post": ComponentSpec.parse(0, "resid_post"),
        "L1_resid_pre": ComponentSpec.parse(1, "resid_pre"),
        f"L{n_layers - 1}_resid_post": ComponentSpec.parse(n_layers - 1, "resid_post"),
    }
    sink_root = os.environ.get("SAE_TEST_SINK") or str(tmp_path_factory.mktemp("sink"))
    sink_path = os.path.join(sink_root, f"vllm-gpu-test-{os.getpid()}")
    extractor = VLLMExtractor(
        model_name=MODEL,
        dtype="bfloat16",
        trust_remote_code=True,
        specs=specs,
        num_hidden_layers=n_layers,
        seq_len=SEQ_LEN,
        vllm_cfg=VLLMCfg(gpu_memory_utilization=0.9, sequences_per_call=N_SEQUENCES),
        # Keep the files so the test can inspect what vLLM wrote.
        sink_cfg=ActivationSinkCfg(path=sink_path, lifecycle="cache"),
        capture_device="cpu",
    )
    token_ids = torch.randint(1_000, 20_000, (N_SEQUENCES, SEQ_LEN), generator=torch.Generator().manual_seed(0))
    batch = extractor.extract(token_ids)
    files = sorted(extractor.sink.path.glob("*.safetensors"))
    raw = load_file(str(files[0])) if files else {}
    print(
        f"\nvllm={vllm.__version__} model={MODEL} layers={n_layers} aux_ids={list(extractor.layout.layer_ids)}"
        f"\nfiles={len(files)} token_ids={tuple(raw['token_ids'].shape) if raw else None}"
        f" hidden_states={tuple(raw['hidden_states'].shape) if raw else None}"
        f" dtype={raw['hidden_states'].dtype if raw else None}"
        f"\npeak_cuda_mem_gib={torch.cuda.max_memory_allocated() / 2**30:.1f} (this process only)"
    )
    yield SimpleNamespace(batch=batch, token_ids=token_ids, files=files, raw=raw, n_layers=n_layers, specs=specs)
    extractor.close()
    shutil.rmtree(sink_path, ignore_errors=True)


def test_one_file_per_sequence_with_the_prompt_tokens(vllm_run):
    assert len(vllm_run.files) == N_SEQUENCES
    assert vllm_run.raw["token_ids"].shape[0] in (SEQ_LEN, SEQ_LEN + 1)
    assert vllm_run.raw["hidden_states"].shape[1] == 3  # ids 0, 1 and n_layers


def test_every_site_has_one_finite_vector_per_token(vllm_run):
    for slug, activations in vllm_run.batch.activations.items():
        assert activations.shape[0] == N_SEQUENCES * SEQ_LEN, slug
        assert torch.isfinite(activations.float()).all(), slug


def test_sites_sharing_a_layer_boundary_are_identical(vllm_run):
    torch.testing.assert_close(vllm_run.batch.activations["L0_resid_post"], vllm_run.batch.activations["L1_resid_pre"])
    assert not torch.equal(vllm_run.batch.activations["L0_resid_pre"], vllm_run.batch.activations["L0_resid_post"])
