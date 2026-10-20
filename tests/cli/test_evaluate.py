import json
import sys

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from sae_pipeline.cache.manifest import validation_dir_for
from sae_pipeline.cli import evaluate
from sae_pipeline.eval.output import evaluation_directory
from sae_pipeline.sae.jumprelu import JumpReLUSAE
from tests.synthetic import write_cache

ARGS = ["--layer", "0", "--component", "resid_post", "--arch", "jumprelu", "--width", "8", "--l0", "2"]


def _config(tmp_path):
    config = tmp_path / "cfg.yaml"
    config.write_text(
        "run_id: run\n"
        "data: {total_tokens: 10}\n"
        "target: {layer: 0, component: resid_post}\n"
        f"cache: {{cache_dir: {(tmp_path / 'caches').as_posix()}}}\n"
        f"sae: {{batch_size: 16, ckpt_dir: {(tmp_path / 'ckpt').as_posix()}}}\n"
        "eval: {fvu_n_tokens: 64, dead_n_tokens: 64}\n"
        f"log: {{log_dir: {(tmp_path / 'logs').as_posix()}}}\n"
    )
    return config


def _silent_sae_checkpoint(tmp_path, mean: torch.Tensor) -> None:
    """An SAE that never fires and reconstructs every token as ``mean``."""
    sae = JumpReLUSAE(d_in=4, d_sae=8)
    with torch.no_grad():
        sae.W_enc.zero_()
        sae.b_enc.zero_()
        sae.b_dec.copy_(mean)
    directory = tmp_path / "ckpt" / "run" / "L0_resid_post" / "jumprelu_w8_l0_2"
    directory.mkdir(parents=True)
    save_file({key: value.contiguous() for key, value in sae.state_dict().items()},
              str(directory / "sae_step_0000001.safetensors"))


@pytest.mark.parametrize("partition", ["train", "validation"])
def test_evaluation_scores_the_latest_checkpoint_on_the_selected_cache(tmp_path, monkeypatch, partition):
    x = torch.randn(64, 4, generator=torch.Generator().manual_seed(0)) * 2 + 1
    if partition == "train":
        write_cache(tmp_path / "caches" / "run" / "L0_resid_post", x)
        selector = ["--partition", "train"]
    else:
        write_cache(validation_dir_for(tmp_path / "caches", "run", "de", "L0_resid_post"), x, language="de")
        selector = ["--partition", "validation", "--language", "de"]
    _silent_sae_checkpoint(tmp_path, x.mean(dim=0))
    monkeypatch.setattr(sys, "argv", ["evaluate", "--config", str(_config(tmp_path)), *ARGS, *selector, "--device", "cpu"])

    evaluate.main()

    out_dir = evaluation_directory(tmp_path / "logs", "run", "L0_resid_post", partition,
                                   "de" if partition == "validation" else None, None)
    summary = json.loads((out_dir / "jumprelu_w8_l0_2_eval.json").read_text())
    assert summary["checkpoint"].endswith("sae_step_0000001.safetensors")
    assert summary["n_tokens"] == 64
    assert summary["l0"] == 0.0 and summary["dead_pct"] == 100.0
    assert summary["fvu"] == pytest.approx(1.0, abs=1e-5)
    arrays = np.load(out_dir / "jumprelu_w8_l0_2_eval_arrays.npz")
    assert arrays["firing_frequency"].shape == (8,)


def test_a_validation_evaluation_needs_a_language(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["evaluate", "--config", str(_config(tmp_path)), *ARGS, "--partition", "validation"])
    with pytest.raises(SystemExit):
        evaluate.main()
    assert "--language is required" in capsys.readouterr().err


def test_delta_ce_flag_scores_a_bounded_heldout_probe(tmp_path, monkeypatch):
    import yaml
    from sae_pipeline.data import streaming
    from sae_pipeline.model import loader
    from tests.eval.test_delta_ce import TinyLM

    config = _config(tmp_path)
    settings = yaml.safe_load(config.read_text())
    settings["eval"].update(delta_ce_n_seqs=2, delta_ce_seq_len=3)
    config.write_text(yaml.safe_dump(settings))
    write_cache(tmp_path / "caches/run/L0_resid_post", torch.randn(64, 4))
    _silent_sae_checkpoint(tmp_path, torch.zeros(4))
    seen = []
    def model_and_tokenizer(cfg):
        seen.append(cfg.device_map)
        return TinyLM(), None
    monkeypatch.setattr(loader, "load_model_and_tokenizer", model_and_tokenizer)
    monkeypatch.setattr(streaming, "make_token_loader", lambda *a, **kw: iter([torch.tensor([[0, 0, 0]])] * 2))
    monkeypatch.setattr(sys, "argv", ["evaluate", "--config", str(config), *ARGS, "--device", "cpu", "--delta-ce"])
    evaluate.main()
    directory = evaluation_directory(tmp_path / "logs", "run", "L0_resid_post", "train", None, None)
    summary = json.loads((directory / "jumprelu_w8_l0_2_eval.json").read_text())
    assert seen == [{"": "cpu"}]
    assert summary["delta_ce"] > 1
    assert summary["delta_ce_n_seqs"] == 2 and summary["delta_ce_seq_len"] == 3
