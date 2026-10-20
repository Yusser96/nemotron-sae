import torch
import pytest
from safetensors.torch import save_file
from sae_pipeline.config import SAECfg
from sae_pipeline.sae.jumprelu import JumpReLUSAE
from sae_pipeline.sae.initialization import initialize_sae, parameter_digest


def fresh():
    torch.manual_seed(42)
    return JumpReLUSAE(8, 16)


def checkpoint(tmp_path):
    torch.manual_seed(7)
    model = JumpReLUSAE(8, 16)
    with torch.no_grad():
        model.b_dec.fill_(0.7)
        model.b_enc.fill_(0.2)
        model.log_theta.fill_(-4.0)
    path = tmp_path / 'warm.safetensors'
    save_file(model.state_dict(), str(path))
    return model, path


def test_partial_reset_preserves_kept_and_reuses_exact_random_columns(tmp_path):
    warm, path = checkpoint(tmp_path)
    random = fresh()
    partial = fresh()
    before_rng = torch.random.get_rng_state().clone()
    cfg = SAECfg(checkpoint_source=path, initialization_mode='partial_reset')
    receipt = initialize_sae(partial, cfg)
    assert torch.equal(before_rng, torch.random.get_rng_state())
    reset = torch.tensor(receipt['reset_indices'])
    assert len(reset) == 8 and len(reset.unique()) == 8
    kept = torch.tensor(sorted(set(range(16)) - set(reset.tolist())))
    for key in ('W_enc', 'W_dec'):
        torch.testing.assert_close(getattr(partial, key)[:, reset], getattr(random, key)[:, reset], rtol=0, atol=0)
        torch.testing.assert_close(getattr(partial, key)[:, kept], getattr(warm, key)[:, kept], rtol=0, atol=0)
    for key in ('b_enc', 'log_theta'):
        torch.testing.assert_close(getattr(partial, key)[reset], getattr(random, key)[reset], rtol=0, atol=0)
        torch.testing.assert_close(getattr(partial, key)[kept], getattr(warm, key)[kept], rtol=0, atol=0)
    torch.testing.assert_close(partial.b_dec, warm.b_dec, rtol=0, atol=0)


def test_random_does_not_access_checkpoint_and_warm_is_unchanged(tmp_path):
    warm, path = checkpoint(tmp_path)
    model = fresh()
    original = parameter_digest(model)
    receipt = initialize_sae(model, SAECfg(initialization_mode='random', checkpoint_source='/not/a/checkpoint'))
    assert receipt['parameter_sha256'] == original
    assert receipt['reset_count'] == 16 and receipt['checkpoint'] is None
    loaded = fresh()
    receipt = initialize_sae(loaded, SAECfg(checkpoint_source=path))
    assert receipt['parameter_sha256'] == parameter_digest(warm)


def test_reset_happens_after_raw_export_unfold(tmp_path):
    model = fresh()
    raw = tmp_path / 'raw.safetensors'
    save_file(model.state_dict(), str(raw))
    warm = fresh()
    warm.unfold_raw_export(2.0)
    partial = fresh()
    receipt = initialize_sae(partial, SAECfg(checkpoint_source=raw, checkpoint_format='raw_export',
                            input_normalization='whole_vector', initialization_mode='partial_reset'),
                             normalization_scale=2.0)
    reset = set(receipt['reset_indices'])
    kept = sorted(set(range(16)) - reset)
    torch.testing.assert_close(partial.W_enc[:, kept], warm.W_enc[:, kept], rtol=0, atol=0)
    torch.testing.assert_close(partial.b_dec, warm.b_dec, rtol=0, atol=0)


def test_checkpoint_required_and_config_rejects_bad_fraction():
    with pytest.raises(ValueError):
        initialize_sae(fresh(), SAECfg(initialization_mode='partial_reset'))
    with pytest.raises(ValueError):
        SAECfg(initialization_reset_fraction=1.0)
