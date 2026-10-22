"""Shared fixtures and fakes for the streaming SAE training tests."""

import dataclasses
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from sae_pipeline.config import EarlyStoppingCfg, SAECfg, StreamCfg
from sae_pipeline.extraction.base import ActivationBatch
from sae_pipeline.sae.jumprelu import JumpReLUSAE
from sae_pipeline.sae.train_streaming import (
    SiteEvaluation,
    evaluate_site,
    init_streaming_site,
    run_streaming_training,
)
from tests.synthetic import write_cache

D_IN, D_SAE, L0_TARGET, BATCH = 6, 16, 3, 32
SEQ_LEN, TOKENS_PER_FWD = 8, 2 * BATCH


def write_warm_start(path: Path) -> Path:
    torch.manual_seed(3)
    sae = JumpReLUSAE(d_in=D_IN, d_sae=D_SAE, bandwidth=0.05)
    save_file({key: value.detach().contiguous() for key, value in sae.state_dict().items()}, str(path))
    return path


def make_sae_cfg(checkpoint: Path, **overrides) -> SAECfg:
    values = dict(
        arch="jumprelu",
        d_sae=D_SAE,
        l0_target=L0_TARGET,
        lr=1.0e-2,
        batch_size=BATCH,
        n_steps=1,
        warmup_steps=2,
        l0_warmup_steps=2,
        bandwidth=0.05,
        lr_schedule="cosine_warmup_constant",
        reconstruction_loss="vector_sum",
        gradient_clip_norm=1.0,
        inverse_simpson_floor=1.0,
        inverse_simpson_max_scale=5.0,
        log_every=1,
        ckpt_every=1,
        n_batches_in_buffer=1,
        checkpoint_source=checkpoint,
        checkpoint_mode="finetune",
    )
    values.update(overrides)
    return SAECfg(**values)


def make_site(tmp_path: Path, cfg: SAECfg, validation_dirs=None, probe_tokens=BATCH, **kwargs):
    return init_streaming_site(
        slug="site",
        sae_cfg=cfg,
        d_in=D_IN,
        arch="jumprelu",
        d_sae=D_SAE,
        l0_target=L0_TARGET,
        out_dir=tmp_path / "streaming",
        device="cpu",
        validation_cache_dirs=validation_dirs or {},
        probe_tokens=probe_tokens,
        **kwargs,
    )


def make_validation_dirs(tmp_path: Path, n_rows: int = 2 * BATCH) -> dict[str, Path]:
    return {
        language: write_cache(
            tmp_path / "validation" / language,
            torch.randn(n_rows, D_IN, generator=torch.Generator().manual_seed(seed)),
            language=language,
        )
        for seed, language in enumerate(("de", "en"))
    }


class CountingStream:
    """Token batches whose first token is a global sequence index."""

    def __init__(self) -> None:
        self.position = 0
        self.first_rows: list[int] = []

    def next_batch(self, batch_size: int) -> torch.Tensor:
        rows = torch.arange(self.position, self.position + batch_size).unsqueeze(1).repeat(1, SEQ_LEN)
        self.first_rows.append(self.position)
        self.position += batch_size
        return rows

    def state_dict(self) -> dict[str, object]:
        return {"position": self.position}

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.position = int(state["position"])


class SeededExtractor:
    """Deterministic activations derived from each batch's first token."""

    def extract(self, token_batch: torch.Tensor) -> ActivationBatch:
        generator = torch.Generator().manual_seed(int(token_batch[0, 0]))
        return ActivationBatch(token_batch, {"site": torch.randn(token_batch.numel(), D_IN, generator=generator)})


class StallingStream(CountingStream):
    """Runs dry after ``batches`` batches, as a fill stream whose producer stopped."""

    def __init__(self, batches: int) -> None:
        super().__init__()
        self.batches = batches

    def next_batch(self, batch_size: int) -> torch.Tensor:
        if len(self.first_rows) == self.batches:
            raise TimeoutError("no fill arrived")
        return super().next_batch(batch_size)


def run_training(tmp_path: Path, site, stream, *, stream_cfg: StreamCfg, early_stopping=None, name=None) -> bool:
    return run_streaming_training(
        sites={"site": site},
        token_stream=stream,
        extractor=SeededExtractor(),
        stream_cfg=stream_cfg,
        early_stopping=early_stopping or EarlyStoppingCfg(),
        tokens_per_fwd=TOKENS_PER_FWD,
        seq_len=SEQ_LEN,
        device="cpu",
        out_root=tmp_path,
        name=name,
    )


def logged_steps(site) -> list[int]:
    return [json.loads(line)["step"] for line in site.log_path.read_text().splitlines()]


def patch_evaluations(monkeypatch, *, l0=None, objective=None):
    """Make every evaluation report this L0 and/or objective."""
    real = evaluate_site

    def changed(site, device):
        evaluation = real(site, device)
        metrics = {k: dataclasses.replace(m, l0=m.l0 if l0 is None else l0) for k, m in evaluation.metrics.items()}
        objectives = {k: v if objective is None else objective for k, v in evaluation.objective.items()}
        return SiteEvaluation(metrics, objectives)

    monkeypatch.setattr("sae_pipeline.sae.train_streaming.evaluate_site", changed)
