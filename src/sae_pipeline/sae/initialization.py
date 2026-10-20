"""Explicit initialization of streaming SAEs; resume is handled separately."""
from __future__ import annotations
import hashlib
import math
import torch
from sae_pipeline.sae.checkpoint import load_sae_weights, resolve_checkpoint


def parameter_digest(sae):
    digest = hashlib.sha256()
    for name, parameter in sorted(sae.state_dict().items()):
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def initialize_sae(sae, cfg, *, normalization_scale=1.0, checkpoint_filename=None):
    """Initialize once, never on resume; the caller has built a fresh optimizer.

    Partial resets use the already constructed random SAE as their replacement
    dictionary. A local RNG picks columns without changing training's RNG state.
    The checkpoint is unfolded into normalized training coordinates first.
    """
    mode = cfg.initialization_mode
    indices = torch.empty(0, dtype=torch.long)
    source = None
    fresh = None
    if mode == "partial_reset":
        if not 0 < cfg.initialization_reset_fraction < 1:
            raise ValueError("Partial reset fraction must be strictly between zero and one")
        fresh = {key: value.detach().clone() for key, value in sae.state_dict().items()}
        count = math.floor(cfg.initialization_reset_fraction * sae.d_sae)
        if count == 0:
            raise ValueError("Partial reset selects no latents")
        indices = torch.randperm(sae.d_sae, generator=torch.Generator().manual_seed(
            cfg.initialization_reset_seed))[:count].sort().values
    if mode != "random":
        if cfg.checkpoint_source is None:
            raise ValueError("Checkpoint and partial-reset initialization require checkpoint_source")
        checkpoint = resolve_checkpoint(cfg.checkpoint_source, filename=checkpoint_filename,
                                        revision=cfg.checkpoint_revision, require_trainer_state=False)
        load_sae_weights(sae, checkpoint.weights_path)
        source = str(checkpoint.weights_path)
        if cfg.checkpoint_format == "raw_export":
            if cfg.input_normalization != "whole_vector":
                raise ValueError("Raw-export initialization requires whole_vector normalization")
            sae.unfold_raw_export(normalization_scale)
    else:
        indices = torch.arange(sae.d_sae)
    if fresh is not None:
        with torch.no_grad():
            selected = indices.to(sae.W_enc.device)
            for key in ("W_enc", "W_dec"):
                getattr(sae, key)[:, selected] = fresh[key][:, selected]
            for key in ("b_enc", "log_theta"):
                getattr(sae, key)[selected] = fresh[key][selected]
    return dict(mode=mode, checkpoint=source, reset_seed=cfg.initialization_reset_seed,
                reset_count=len(indices), reset_indices=indices.tolist(),
                preserved_global_decoder_bias=mode != "random", optimizer="fresh",
                parameter_sha256=parameter_digest(sae),
                threshold_policy="factory_0.001_for_reset_features")
