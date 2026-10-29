"""Train a JumpReLU SAE on synthetic activations built from a known sparse dictionary, on a CPU, in under a minute.

Each token mixes three of 64 ground-truth directions in a 32-dimensional space. The script writes the activations
as a cache, trains with the cached trainer, then reports reconstruction quality, sparsity and how many
ground-truth directions the decoder recovered.

    python examples/toy_sae.py [OUT_DIR]
"""

import sys
from pathlib import Path

import torch

from sae_pipeline.cache.manifest import CacheManifest
from sae_pipeline.cache.writer import ShardWriter
from sae_pipeline.config import SAECfg
from sae_pipeline.sae.checkpoint import load_sae_weights
from sae_pipeline.sae.jumprelu import JumpReLUSAE
from sae_pipeline.sae.train import train_sae

D_IN, N_TRUE, ACTIVE, TOKENS = 32, 64, 3, 65_536


def synthetic(n, dictionary, generator):
    """Tokens that are sums of ACTIVE dictionary rows with positive coefficients."""
    rows = torch.rand(n, N_TRUE, generator=generator).topk(ACTIVE, dim=1).indices
    codes = torch.zeros(n, N_TRUE).scatter_(1, rows, 1 + torch.rand(n, ACTIVE, generator=generator))
    return codes @ dictionary


def write_cache(path, x):
    manifest = CacheManifest(run_id="toy", model="toy", dtype="float32", layer=0, component="resid_post",
                             d_activation=D_IN, partition="train")
    with ShardWriter(out_dir=path, d_activation=D_IN, shard_size_bytes=1 << 20, dtype=torch.float32,
                     shuffle_seed=0, manifest=manifest) as writer:
        writer.add(x)
    writer.manifest.complete = True
    writer.manifest.write(path / "manifest.json")


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/toy_sae")
    generator = torch.Generator().manual_seed(0)
    dictionary = torch.nn.functional.normalize(torch.randn(N_TRUE, D_IN, generator=generator), dim=1)
    write_cache(out / "cache", synthetic(TOKENS, dictionary, generator))

    # The JumpReLU threshold learns through a straight-through window of width `bandwidth`; at this activation
    # scale (about 1) the default 0.001 gives the threshold almost no gradient, so the toy widens it.
    cfg = SAECfg(d_sae=N_TRUE, l0_target=ACTIVE, initialization_mode="random", lr=1e-2, batch_size=256,
                 bandwidth=0.2, n_steps=10_000, warmup_steps=100, l0_warmup_steps=500, ckpt_every=10_000,
                 log_every=1_000)
    checkpoint = train_sae(cfg=cfg, cache_dir=out / "cache", d_in=D_IN, arch="jumprelu", d_sae=N_TRUE,
                           l0_target=ACTIVE, out_dir=out / "sae", device="cpu")

    sae = JumpReLUSAE(d_in=D_IN, d_sae=N_TRUE)
    load_sae_weights(sae, checkpoint)
    x = synthetic(8_192, dictionary, generator)
    with torch.no_grad():
        features = sae.encode(x)
        error = sae.decode(features) - x
    fvu = float(error.square().sum() / (x - x.mean(0)).square().sum())
    l0 = float((features > 0).float().sum(1).mean())
    # A ground-truth direction counts as recovered when some decoder direction points the same way (cosine > 0.9).
    # The decoder is stored as [d_in, d_sae]: one column per latent.
    similarity = torch.nn.functional.normalize(sae.W_dec.detach().T, dim=1) @ dictionary.T
    recovered = int((similarity.max(0).values > 0.9).sum())
    print(f"checkpoint: {checkpoint}")
    print(f"FVU {fvu:.3f}, L0 {l0:.2f} (target {ACTIVE}), recovered {recovered}/{N_TRUE} directions")


if __name__ == "__main__":
    main()
