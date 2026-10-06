import pytest
import torch
from safetensors.torch import save_file

from sae_pipeline.cli.compare_sae_metrics import compare_sae_metrics, load_raw_sae
from sae_pipeline.sae.jumprelu import JumpReLUSAE


def _sae_file(path, d_in=6, d_sae=12):
    torch.manual_seed(0)
    sae = JumpReLUSAE(d_in=d_in, d_sae=d_sae, pre_encoder_bias=False)
    save_file({key: value.detach().contiguous() for key, value in sae.state_dict().items()}, str(path))
    return path


def _dump(path, tokens, **sites):
    save_file({"token_ids": tokens, **sites}, str(path))
    return path


def test_loaded_sae_takes_its_shape_from_the_weights(tmp_path):
    sae = load_raw_sae(_sae_file(tmp_path / "sae.safetensors", d_in=6, d_sae=12))

    assert (sae.d_in, sae.d_sae) == (6, 12)
    assert sae.pre_encoder_bias is False


def test_identical_dumps_give_zero_deltas_and_noise_gives_nonzero(tmp_path):
    torch.manual_seed(1)
    tokens = torch.arange(64).reshape(4, 16)
    x = torch.randn(64, 6)
    weights = _sae_file(tmp_path / "sae.safetensors")
    reference = _dump(tmp_path / "a.safetensors", tokens, L1=x)
    same = _dump(tmp_path / "b.safetensors", tokens, L1=x.clone())
    noisy = _dump(tmp_path / "c.safetensors", tokens, L1=x + 0.5 * torch.randn(64, 6))

    exact = compare_sae_metrics(reference, same, {"L1": weights}, batch_size=16)["L1"]
    perturbed = compare_sae_metrics(reference, noisy, {"L1": weights}, batch_size=16)["L1"]

    assert all(value == 0.0 for value in exact["delta"].values())
    assert exact["reference"]["fvu"] == pytest.approx(exact["candidate"]["fvu"])
    assert perturbed["delta"]["fvu"] != 0.0


def test_a_site_missing_from_a_dump_is_an_error(tmp_path):
    tokens = torch.arange(4).reshape(1, 4)
    reference = _dump(tmp_path / "a.safetensors", tokens, L1=torch.randn(4, 6))
    candidate = _dump(tmp_path / "b.safetensors", tokens, L2=torch.randn(4, 6))

    with pytest.raises(KeyError, match="L1"):
        compare_sae_metrics(reference, candidate, {"L1": _sae_file(tmp_path / "sae.safetensors")})
