"""Streaming SAE run loop: token caps, pauses, fills, checkpoints and normalisation metadata."""

import dataclasses
import itertools
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from sae_pipeline.config import EarlyStoppingCfg, StreamCfg
from sae_pipeline.extraction.base import ActivationBatch
from sae_pipeline.extraction.fills import FillQueue, FillStream, fill_dir, write_header
from sae_pipeline.sae.checkpoint_source import resolve_checkpoint
from sae_pipeline.sae.train_streaming import (
    evaluate_and_update,
    init_streaming_site,
    run_streaming_training,
    write_checkpoint,
)
from tests.sae.streaming_fakes import (
    BATCH,
    D_IN,
    D_SAE,
    L0_TARGET,
    SEQ_LEN,
    TOKENS_PER_FWD,
    CountingStream,
    SeededExtractor,
    StallingStream,
    logged_steps,
    make_sae_cfg,
    make_site,
    make_validation_dirs,
    run_training,
    write_warm_start,
)


def test_runs_to_the_token_cap_with_scheduled_evaluations(tmp_path):
    cfg = make_sae_cfg(write_warm_start(tmp_path / "warm.safetensors"), eval_interval_steps=2, ckpt_every=2)
    site = make_site(tmp_path, cfg, make_validation_dirs(tmp_path))
    stream = CountingStream()

    done = run_training(tmp_path, site, stream, stream_cfg=StreamCfg(enabled=True, token_cap=6 * BATCH))
    site.close()

    assert done is True
    assert (tmp_path / "DONE").exists()
    assert site.step == 6 and site.stopped
    assert logged_steps(site) == [1, 2, 3, 4, 5, 6]
    evaluated = [json.loads(line)["step"] for line in site.validation_log_path.read_text().splitlines()]
    assert evaluated == [2, 2, 4, 4, 6, 6]
    assert stream.position == 3 * TOKENS_PER_FWD // SEQ_LEN


def test_pauses_at_the_time_limit_and_resumes_without_repeating_tokens(tmp_path, monkeypatch):
    warm = write_warm_start(tmp_path / "warm.safetensors")
    clock = itertools.count()
    monkeypatch.setattr(
        "sae_pipeline.sae.train_streaming.time", SimpleNamespace(monotonic=lambda: float(next(clock)))
    )
    site = make_site(tmp_path, make_sae_cfg(warm, ckpt_every=100))
    first = CountingStream()

    done = run_training(tmp_path, site, first, stream_cfg=StreamCfg(enabled=True, stop_after_seconds=3))
    site.close()

    assert done is False and not (tmp_path / "DONE").exists()
    assert site.step == 4  # two forward passes, each two SAE batches

    resumed = make_site(
        tmp_path,
        make_sae_cfg(warm, ckpt_every=100),
        resume_checkpoint=resolve_checkpoint(site.out_dir, require_trainer_state=True),
    )
    second = CountingStream()
    done = run_training(tmp_path, resumed, second, stream_cfg=StreamCfg(enabled=True, token_cap=8 * BATCH))
    resumed.close()

    assert done is True
    assert resumed.step == 8
    assert logged_steps(resumed) == list(range(1, 9))
    assert set(first.first_rows).isdisjoint(second.first_rows)
    assert second.first_rows[0] == first.position


def test_a_named_trainer_pauses_when_its_stream_times_out_and_resumes_after_it(tmp_path):
    warm = write_warm_start(tmp_path / "warm.safetensors")
    site = make_site(tmp_path, make_sae_cfg(warm, ckpt_every=100))
    first = StallingStream(batches=2)

    done = run_training(tmp_path, site, first, stream_cfg=StreamCfg(enabled=True), name="c0")
    site.close()

    assert done is False and site.step == 4
    assert (tmp_path / "stream_state_c0.pt").exists() and not (tmp_path / "stream_state.pt").exists()
    resumed = make_site(
        tmp_path,
        make_sae_cfg(warm, ckpt_every=100),
        resume_checkpoint=resolve_checkpoint(site.out_dir, require_trainer_state=True),
    )
    second = CountingStream()
    done = run_training(tmp_path, resumed, second, stream_cfg=StreamCfg(enabled=True, token_cap=6 * BATCH), name="c0")
    resumed.close()

    assert done is True and (tmp_path / "DONE_c0").exists() and not (tmp_path / "DONE").exists()
    assert second.first_rows[0] == first.position


def test_two_consumers_train_their_own_sites_from_one_producers_fills(tmp_path):
    fills = tmp_path / "fills"
    write_header(fills, {"model": "m", "seq_len": SEQ_LEN, "slugs": ["a", "b"]})
    generator = torch.Generator().manual_seed(0)
    for index in range(3):
        partial = fills / f"fill_{index:06d}.partial"
        partial.mkdir()
        save_file({"token_ids": torch.zeros(6, SEQ_LEN, dtype=torch.long)}, str(partial / "token_ids.safetensors"))
        for slug in ("a", "b"):
            save_file({"activations": torch.randn(6 * SEQ_LEN, D_IN, generator=generator)}, str(partial / f"{slug}.safetensors"))
        partial.rename(fill_dir(fills, index))
    warm = write_warm_start(tmp_path / "warm.safetensors")
    run = tmp_path / "run"

    for name, slug in (("c0", "a"), ("c1", "b")):
        site = init_streaming_site(
            slug=slug, sae_cfg=make_sae_cfg(warm), d_in=D_IN, arch="jumprelu", d_sae=D_SAE, l0_target=L0_TARGET,
            out_dir=run / slug, device="cpu", validation_cache_dirs={}, probe_tokens=BATCH,
        )
        stream = FillStream(fills, name, [slug], header={"model": "m", "seq_len": SEQ_LEN}, device="cpu", poll_seconds=0)
        done = run_streaming_training(
            sites={slug: site}, token_stream=stream, extractor=stream,
            stream_cfg=StreamCfg(enabled=True, token_cap=4 * BATCH), early_stopping=EarlyStoppingCfg(),
            tokens_per_fwd=TOKENS_PER_FWD, seq_len=SEQ_LEN, device="cpu", out_root=run, name=name,
        )
        stream.close()
        site.close()
        assert done and site.step == 4
        # Two forward passes of 8 sequences: fills 0 and 1, and 4 sequences of fill 2.
        assert stream.state_dict() == {"fill": 2, "sequence": 4}

    assert sorted(p.name for p in run.iterdir()) == ["DONE_c0", "DONE_c1", "a", "b", "stream_state_c0.pt", "stream_state_c1.pt"]
    assert FillQueue(fills, ["c0", "c1"]).release() == 2


def test_normalized_streaming_checkpoints_and_best_carry_metadata(tmp_path):
    from sae_pipeline.sae.normalization import (
        WholeVectorNormalization,
        write_normalization,
    )
    warm = write_warm_start(tmp_path / "warm.safetensors")
    metadata = tmp_path / "norm"
    metadata.mkdir()
    statistic = WholeVectorNormalization(0.5, 4.0, 64, D_IN)
    write_normalization(metadata / "normalization.json", statistic)
    cfg = make_sae_cfg(warm, input_normalization="whole_vector")
    validation = make_validation_dirs(tmp_path)
    site = make_site(tmp_path, cfg, validation, normalization_dir=metadata)
    site.step = 50000
    evaluate_and_update(site, 50000, "cpu", EarlyStoppingCfg(divergence_l0_factor=None))
    write_checkpoint(site, 50000)
    site.close()
    expected = json.loads((metadata / "normalization.json").read_text())
    assert json.loads((site.out_dir / "normalization.json").read_text()) == expected
    assert json.loads((site.out_dir / "best" / "normalization.json").read_text()) == expected
    resumed = make_site(tmp_path, cfg, validation, resume_checkpoint=resolve_checkpoint(site.out_dir, require_trainer_state=True))
    assert resumed.normalization_scale == 0.5
    resumed.close()
    write_normalization(metadata / "normalization.json", dataclasses.replace(statistic, input_scale=0.25))
    with pytest.raises(ValueError, match="normalization scale differs"):
        make_site(tmp_path, cfg, validation, normalization_dir=metadata,
              resume_checkpoint=resolve_checkpoint(site.out_dir, require_trainer_state=True))
    assert json.loads((site.out_dir / "normalization.json").read_text()) == expected


def test_token_cap_stops_inside_forward_and_requires_batch_alignment(tmp_path, monkeypatch):
    monkeypatch.setattr("sae_pipeline.sae.train_streaming.refresh_plot", lambda *a, **kw: None)
    warm = write_warm_start(tmp_path / "warm.safetensors")
    site = make_site(tmp_path, make_sae_cfg(warm, ckpt_every=3))
    with pytest.raises(ValueError, match="multiple"):
        run_training(tmp_path, site, CountingStream(), stream_cfg=StreamCfg(token_cap=BATCH + 1))
    assert site.step == 0
    assert run_training(tmp_path, site, CountingStream(), stream_cfg=StreamCfg(token_cap=3 * BATCH))
    site.close()
    assert site.step == 3 and site.tokens_seen() == 3 * BATCH


def test_periodic_snapshot_is_taken_after_all_batches_in_forward(tmp_path, monkeypatch):
    from sae_pipeline.sae.checkpoint import load_trainer_state
    from sae_pipeline.sae.train_streaming import committed_stream_checkpoints
    monkeypatch.setattr("sae_pipeline.sae.train_streaming.refresh_plot", lambda *a, **kw: None)
    site = make_site(tmp_path, make_sae_cfg(write_warm_start(tmp_path / "warm.safetensors"), ckpt_every=3))
    # The periodic save is triggered at step 3 inside forward 2, but must contain step 4.
    assert not run_training(tmp_path, site, StallingStream(2), stream_cfg=StreamCfg())
    site.close()
    committed = committed_stream_checkpoints(tmp_path, {"site": site.out_dir})["site"]
    state = load_trainer_state(tmp_path / "stream_state.pt")
    assert committed.step == 4
    assert state["tokens_processed"] == 4 * BATCH


@pytest.mark.parametrize("failure_point", ["second_site", "cursor"])
def test_interrupted_multi_site_snapshot_keeps_previous_pairs_and_resumes_exactly(tmp_path, monkeypatch, failure_point):
    import sae_pipeline.sae.train_streaming as module
    monkeypatch.setattr(module, "refresh_plot", lambda *a, **kw: None)
    warm = write_warm_start(tmp_path / "warm.safetensors")
    cfg = make_sae_cfg(warm, ckpt_every=3, keep_last_checkpoints=1)
    class MultiExtractor:
        def extract(self, tokens):
            batch = SeededExtractor().extract(tokens)
            return ActivationBatch(tokens, {slug: batch.activations["site"] for slug in ("a", "b")})
    def make_sites(root, resume=False):
        selected = module.committed_stream_checkpoints(root, {s: root / s for s in ("a", "b")}) if resume else {}
        return {s: init_streaming_site(slug=s, sae_cfg=cfg, d_in=D_IN, arch="jumprelu", d_sae=D_SAE,
                    l0_target=L0_TARGET, out_dir=root / s, device="cpu", validation_cache_dirs={},
                    probe_tokens=BATCH, resume_checkpoint=selected.get(s)) for s in ("a", "b")}
    def run(root, sites, stream, cap=None):
        return run_streaming_training(sites=sites, token_stream=stream, extractor=MultiExtractor(),
                stream_cfg=StreamCfg(token_cap=cap), early_stopping=EarlyStoppingCfg(),
                tokens_per_fwd=TOKENS_PER_FWD, seq_len=SEQ_LEN, device="cpu", out_root=root)
    root = tmp_path / "interrupted"
    sites = make_sites(root)
    assert not run(root, sites, StallingStream(2))
    for site in sites.values(): site.close()
    old = module.committed_stream_checkpoints(root, {s: root / s for s in sites})
    sites = make_sites(root, resume=True)
    with monkeypatch.context() as fault:
        if failure_point == "second_site":
            actual = module.write_checkpoint
            def failing(site, step, **kwargs):
                if site.slug == "b": raise RuntimeError("interrupted transaction")
                return actual(site, step, **kwargs)
            fault.setattr(module, "write_checkpoint", failing)
        else:
            fault.setattr(module, "_save_stream_state", lambda *a: (_ for _ in ()).throw(RuntimeError("interrupted transaction")))
        with pytest.raises(RuntimeError, match="interrupted transaction"):
            run(root, sites, CountingStream(), 8 * BATCH)
    for site in sites.values(): site.close()
    for checkpoint in old.values():
        assert checkpoint.weights_path.exists() and checkpoint.trainer_state_path.exists()
    assert module.committed_stream_checkpoints(root, {s: root / s for s in sites}) == old
    resumed = make_sites(root, resume=True)
    assert run(root, resumed, CountingStream(), 8 * BATCH)
    reference_root = tmp_path / "reference"
    reference = make_sites(reference_root)
    assert run(reference_root, reference, CountingStream(), 8 * BATCH)
    for slug in resumed:
        resumed[slug].close(); reference[slug].close()
        for key, weights in reference[slug].sae.state_dict().items():
            torch.testing.assert_close(resumed[slug].sae.state_dict()[key], weights, rtol=0, atol=0)
        assert logged_steps(resumed[slug]) == list(range(1, 9))

