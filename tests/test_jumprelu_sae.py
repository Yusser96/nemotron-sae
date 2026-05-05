"""Sanity-check JumpReLU SAE: shapes, gradients, sparsity, decoder norm invariance."""

import math

import torch

from sae_pipeline.sae.jumprelu import JumpReLUSAE


def test_jumprelu_forward_shape():
    d_in, d_sae = 32, 256
    sae = JumpReLUSAE(d_in=d_in, d_sae=d_sae)
    x = torch.randn(64, d_in)
    x_hat, f = sae(x)
    assert x_hat.shape == (64, d_in)
    assert f.shape == (64, d_sae)


def test_jumprelu_decoder_unit_norm_after_renorm():
    sae = JumpReLUSAE(d_in=16, d_sae=128)
    sae.W_dec.data.mul_(3.7)  # break unit norm
    sae.renormalize_decoder()
    norms = sae.W_dec.data.norm(dim=0)
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)


def test_jumprelu_l0_decreases_with_higher_threshold():
    sae = JumpReLUSAE(d_in=16, d_sae=128)
    x = torch.randn(256, 16)
    sae.theta.data.fill_(-1.0)   # almost everything fires
    l0_low = sae.hard_l0(x).item()
    sae.theta.data.fill_(5.0)    # almost nothing fires
    l0_high = sae.hard_l0(x).item()
    assert l0_low > l0_high


def test_jumprelu_backward_runs():
    """One step of gradient descent should reduce reconstruction loss on a fixed batch."""
    sae = JumpReLUSAE(d_in=16, d_sae=64, bandwidth=0.01)
    optim = torch.optim.Adam(sae.parameters(), lr=1e-2, betas=(0.0, 0.999))
    x = torch.randn(64, 16)

    loss_history = []
    for _ in range(50):
        x_hat, _ = sae(x)
        loss = (x - x_hat).pow(2).mean()
        optim.zero_grad()
        loss.backward()
        sae.project_decoder_grad()
        optim.step()
        sae.renormalize_decoder()
        loss_history.append(loss.item())

    assert loss_history[-1] < loss_history[0], (loss_history[0], loss_history[-1])


def test_jumprelu_l0_grad_flows_to_theta():
    sae = JumpReLUSAE(d_in=16, d_sae=64, bandwidth=0.05)
    x = torch.randn(64, 16)
    # Encourage a lot of latents to be near the threshold edge so STE has signal.
    sae.theta.data.fill_(0.0)
    l0 = sae.l0(x)
    l0.backward()
    assert sae.theta.grad is not None
    assert sae.theta.grad.abs().sum().item() > 0.0
