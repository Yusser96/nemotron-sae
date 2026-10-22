import pytest
import torch
from safetensors.torch import save_file

from sae_pipeline.cli.compare_activations import compare_dumps


def _dump(path, token_ids, **sites):
    save_file({"token_ids": token_ids, **sites}, str(path), metadata={"backend": path.stem})
    return path


def test_matching_dumps_pass_and_report_every_shared_site(tmp_path):
    torch.manual_seed(0)
    tokens = torch.arange(8).reshape(2, 4)
    a, b = torch.randn(8, 16), torch.randn(8, 16)
    reference = _dump(tmp_path / "hf.safetensors", tokens, L1=a, L2=b, extra=a.clone())
    candidate = _dump(tmp_path / "vllm.safetensors", tokens, L1=a.to(torch.bfloat16), L2=b)

    report = compare_dumps(reference, candidate)

    assert report["passes"] is False
    assert compare_dumps(reference, candidate, allow_missing_sites=True)["passes"] is True
    assert sorted(report["sites"]) == ["L1", "L2"]
    assert report["only_in_reference"] == ["extra"]
    assert report["metadata"]["candidate"] == {"backend": "vllm"}


def test_a_mismatching_site_fails_the_gate(tmp_path):
    tokens = torch.arange(8).reshape(2, 4)
    reference = _dump(tmp_path / "hf.safetensors", tokens, L1=torch.randn(8, 16))
    candidate = _dump(tmp_path / "vllm.safetensors", tokens, L1=torch.randn(8, 16))

    report = compare_dumps(reference, candidate)

    assert report["passes"] is False
    assert report["sites"]["L1"]["passes"] is False


def test_dumps_of_different_tokens_are_rejected(tmp_path):
    reference = _dump(tmp_path / "hf.safetensors", torch.zeros(1, 4, dtype=torch.long), L1=torch.ones(4, 2))
    candidate = _dump(tmp_path / "vllm.safetensors", torch.ones(1, 4, dtype=torch.long), L1=torch.ones(4, 2))

    with pytest.raises(ValueError, match="different token ids"):
        compare_dumps(reference, candidate)
