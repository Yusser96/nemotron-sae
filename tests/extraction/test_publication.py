"""Production registration consumes writer proof without rescanning activations."""

import hashlib
import pytest
import torch
from safetensors.torch import save_file
from sae_pipeline.extraction.publication import entry_from_receipt


def sample(tmp_path):
    ids = torch.tensor([[1, 2]])
    tensors = {"token_ids": ids, "L2_resid_post": torch.ones(2, 3)}
    files = {}
    for key, tensor in tensors.items():
        path = tmp_path / (key + ".safetensors")
        save_file({key: tensor}, str(path))
        files[key] = dict(
            path=str(path),
            shape=list(tensor.shape),
            bytes=path.stat().st_size,
            sha256=hashlib.sha256(
                memoryview(tensor.view(torch.uint8).numpy())
            ).hexdigest(),
        )
    receipt = dict(
        version=2,
        finite_validation="device",
        files=files,
        path=str(tmp_path),
        bytes=sum(v["bytes"] for v in files.values()),
    )
    settings = dict(
        source_digest="frozen",
        sites={"site": dict(layer=2, kind="resid_post", width=3)},
    )
    return receipt, settings, ids


def test_registration_uses_writer_hashes_without_activation_readback(
    tmp_path, monkeypatch
):
    receipt, settings, ids = sample(tmp_path)

    def forbidden(*a, **kw):
        raise AssertionError("Registration must not reopen tensor payloads")

    monkeypatch.setattr("sae_pipeline.extraction.publication.safe_open", forbidden)
    entry = entry_from_receipt(receipt, settings, ids, 0, 1)
    assert entry["format"] == "per_site_v2"
    assert entry["site_sha256"]["site"] == receipt["files"]["L2_resid_post"]["sha256"]


@pytest.mark.parametrize("fault", ["tokens", "width", "validation", "path"])
def test_registration_rejects_invalid_production_contract(tmp_path, fault):
    receipt, settings, ids = sample(tmp_path)
    if fault == "tokens":
        receipt["files"]["token_ids"]["sha256"] = "bad"
    elif fault == "width":
        receipt["files"]["L2_resid_post"]["shape"] = [2, 4]
    elif fault == "validation":
        receipt["finite_validation"] = "unknown"
    else:
        receipt["files"]["L2_resid_post"]["path"] = str(
            tmp_path.parent / "outside.safetensors"
        )
    with pytest.raises(ValueError):
        entry_from_receipt(receipt, settings, ids, 0, 1)
