"""Real-engine check of the vLLM hook extraction (needs vLLM, a GPU and a NemotronH model).

Run with ``SAE_TEST_MODEL=<hf id or path> pytest -m cuda tests/test_vllm_extraction_gpu.py -s``.
``SAE_TEST_SINK`` optionally points the sink somewhere specific (e.g. /dev/shm).
Layers 0, 1 and 5 of NemotronH are a Mamba, a MoE and an attention block.
"""

import os
from types import SimpleNamespace

import pytest
import torch

from sae_pipeline.config import ActivationSinkCfg, VLLMCfg
from sae_pipeline.hooks.components import ComponentSpec

pytestmark = [pytest.mark.cuda, pytest.mark.slow]

MODEL = os.environ.get("SAE_TEST_MODEL")
SEQ_LEN, N_SEQUENCES = 64, 4
SITES = {
    "L0_resid_pre": (0, "resid_pre"),
    "L0_mamba_out": (0, "mamba_out"),
    "L0_resid_post": (0, "resid_post"),
    "L1_resid_pre": (1, "resid_pre"),
    "L1_moe_out": (1, "moe_out"),
    "L5_attn_out_prelinear": (5, "attn_out_prelinear"),
}


@pytest.fixture(scope="module")
def vllm_run(tmp_path_factory):
    vllm = pytest.importorskip("vllm")
    if MODEL is None or not torch.cuda.is_available():
        pytest.skip("needs SAE_TEST_MODEL and a CUDA device")
    from sae_pipeline.extraction.vllm import VLLMExtractor

    sink_root = os.environ.get("SAE_TEST_SINK") or str(tmp_path_factory.mktemp("sink"))
    extractor = VLLMExtractor(
        model_name=MODEL,
        dtype="bfloat16",
        trust_remote_code=True,
        specs={slug: ComponentSpec.parse(layer, kind) for slug, (layer, kind) in SITES.items()},
        seq_len=SEQ_LEN,
        vllm_cfg=VLLMCfg(gpu_memory_utilization=0.9, sequences_per_call=2),
        sink_cfg=ActivationSinkCfg(path=os.path.join(sink_root, f"vllm-gpu-test-{os.getpid()}")),
    )
    token_ids = torch.randint(1_000, 20_000, (N_SEQUENCES, SEQ_LEN), generator=torch.Generator().manual_seed(0))
    batch = extractor.extract(token_ids)
    leftovers = list(extractor.sink.path.iterdir())
    print(f"\nvllm={vllm.__version__} model={MODEL} shapes={ {s: tuple(a.shape) for s, a in batch.activations.items()} }")
    extractor.close()
    yield SimpleNamespace(batch=batch, leftovers=leftovers, sink_path=extractor.sink.path)


def test_every_site_has_one_finite_vector_per_token_and_files_are_consumed(vllm_run):
    for slug, activations in vllm_run.batch.activations.items():
        assert activations.shape == (N_SEQUENCES * SEQ_LEN, 2688), slug
        assert torch.isfinite(activations.float()).all(), slug
    assert vllm_run.leftovers == []
    assert not vllm_run.sink_path.exists()


def test_residual_stream_is_layer_input_plus_mixer_output(vllm_run):
    acts = vllm_run.batch.activations
    torch.testing.assert_close(acts["L0_resid_post"], acts["L1_resid_pre"])
    torch.testing.assert_close(acts["L0_resid_post"], acts["L0_resid_pre"] + acts["L0_mamba_out"])
