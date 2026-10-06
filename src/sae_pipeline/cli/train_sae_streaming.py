"""Streaming SAE training: live model forward passes, no training activation cache.

Trains every site in ``target.sites`` from one continuously running forward
pass over fresh documents (see ``sae.train_streaming``). The only cache it
reads is a fixed per-language validation set, built beforehand with
``cli.cache_activations --partition validation`` under ``validation_run_id``.

Usage:
    python -m sae_pipeline.cli.train_sae_streaming --config <config.yml>

Exits 0 both when training is complete (a ``DONE`` marker exists under the
run's checkpoint root) and when it paused at ``stream.stop_after_seconds``;
rerun the same command to resume.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from sae_pipeline.cache.manifest import CacheManifest, validation_dir_for, validation_manifest_path_for
from sae_pipeline.config import PipelineCfg
from sae_pipeline.data.streaming import ResumableTokenStream
from sae_pipeline.extraction.base import ActivationExtractor
from sae_pipeline.extraction.hf import HFHookExtractor
from sae_pipeline.extraction.vllm_layout import require_residual_site
from sae_pipeline.hooks.components import site_requests
from sae_pipeline.model.loader import load_model_and_tokenizer, load_tokenizer
from sae_pipeline.model.topology import dump_topology, enumerate_hooks
from sae_pipeline.sae.checkpoint import resolve_checkpoint
from sae_pipeline.sae.train_streaming import init_streaming_site, run_streaming_training

log = logging.getLogger(__name__)


def _validation_cache_dirs(cfg: PipelineCfg, target, languages: list[str]) -> tuple[dict[str, Path], int]:
    """Check each language's validation manifest for this site; return its dirs and d_in."""
    validation_run_id = cfg.validation_run_id or cfg.run_id
    dirs: dict[str, Path] = {}
    d_in: int | None = None
    for language in languages:
        manifest_path = validation_manifest_path_for(cfg.cache.cache_dir, validation_run_id, language, target.slug)
        if not manifest_path.is_file():
            raise SystemExit(f"Missing validation manifest: {manifest_path}")
        manifest = CacheManifest.read(manifest_path)
        problems = []
        if not manifest.complete:
            problems.append("incomplete")
        if manifest.model != cfg.model.name:
            problems.append(f"model={manifest.model!r}")
        if (manifest.layer, manifest.component) != (target.layer, target.component):
            problems.append(f"site=({manifest.layer},{manifest.component})")
        if manifest.language != language:
            problems.append(f"language={manifest.language!r}")
        if manifest.total_tokens < cfg.eval.trajectory_n_tokens:
            problems.append(f"tokens={manifest.total_tokens} < eval.trajectory_n_tokens")
        if d_in is not None and manifest.d_activation != d_in:
            problems.append(f"d_activation={manifest.d_activation} differs across languages")
        if problems:
            raise SystemExit(f"Invalid validation cache {manifest_path}: " + "; ".join(problems))
        dirs[language] = validation_dir_for(cfg.cache.cache_dir, validation_run_id, language, target.slug)
        d_in = manifest.d_activation
    assert d_in is not None
    return dirs, d_in


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = PipelineCfg.from_yaml(args.config)
    if not cfg.stream.enabled:
        raise SystemExit("stream.enabled must be true to use train_sae_streaming")
    if os.environ.get("TRAIN_STREAM_STOP_AFTER_SECONDS"):
        cfg = cfg.with_overrides(
            **{"stream.stop_after_seconds": float(os.environ["TRAIN_STREAM_STOP_AFTER_SECONDS"])}
        )
    if not cfg.target.sites:
        raise SystemExit("Streaming training requires target.sites")
    languages = sorted({source.language for source in (cfg.data.sources or []) if source.language})
    if not languages:
        raise SystemExit("Streaming training needs data.sources with a language for each validation set")

    out_root = Path(cfg.sae.ckpt_dir) / cfg.run_id
    if (out_root / "DONE").exists():
        log.info("%s is already complete", out_root)
        return

    arch = cfg.sae.arch if isinstance(cfg.sae.arch, str) else cfg.sae.arch[0]
    d_sae = cfg.sae.d_sae if isinstance(cfg.sae.d_sae, int) else cfg.sae.d_sae[0]
    l0_target = cfg.sae.l0_target if isinstance(cfg.sae.l0_target, int) else cfg.sae.l0_target[0]

    requests = site_requests(cfg.target.sites)
    # Check every validation cache before spending minutes loading the model.
    validation = {target.slug: _validation_cache_dirs(cfg, target, languages) for target in cfg.target.sites}
    use_vllm = cfg.extraction.backend == "vllm"
    extractor: ActivationExtractor
    if use_vllm:
        from sae_pipeline.extraction.vllm import VLLMExtractor

        # Fail before loading anything if a site is not on the residual stream.
        for spec, _site in requests.values():
            require_residual_site(spec)
        tokenizer = load_tokenizer(cfg.model)
        # Start the engine before the SAEs allocate: vLLM sizes itself from
        # extraction.vllm.gpu_memory_utilization at startup.
        specs = {slug: spec for slug, (spec, _site) in requests.items()}
        extractor = VLLMExtractor.from_config(cfg, specs, capture_device=cfg.stream.sae_device)
    else:
        # Pin the whole model to one device: with several GPUs visible, a plain
        # "auto" device_map would spread its layers across all of them.
        model_cfg = cfg.model.model_copy(update={"device_map": {"": cfg.stream.model_device}})
        model, tokenizer = load_model_and_tokenizer(model_cfg)
        model.eval()
        dump_topology(enumerate_hooks(model), out_root / "model_topology.json")
        extractor = HFHookExtractor(
            model, requests, model_device=cfg.stream.model_device, capture_device=cfg.stream.sae_device
        )

    sites = {}
    try:
        for target in cfg.target.sites:
            validation_dirs, d_in = validation[target.slug]
            site_dir = out_root / target.slug
            resume_checkpoint = None
            if site_dir.exists():
                try:
                    resume_checkpoint = resolve_checkpoint(site_dir, require_trainer_state=True)
                except FileNotFoundError:
                    log.warning(
                        "%s: no complete checkpoint pair in %s; starting from the warm start "
                        "(its logs are overwritten)",
                        target.slug,
                        site_dir,
                    )
            checkpoint_filename = cfg.sae.checkpoint_filename
            if cfg.sae.checkpoint_filename_template is not None:
                checkpoint_filename = cfg.sae.checkpoint_filename_template.format(slug=target.slug)
            # sae.normalization_dir holds one precomputed statistic per site slug.
            normalization_dir = None
            if cfg.sae.normalization_dir is not None:
                normalization_dir = Path(cfg.sae.normalization_dir) / target.slug
            sites[target.slug] = init_streaming_site(
                slug=target.slug,
                sae_cfg=cfg.sae,
                d_in=d_in,
                arch=arch,
                d_sae=d_sae,
                l0_target=l0_target,
                out_dir=site_dir,
                device=cfg.stream.sae_device,
                validation_cache_dirs=validation_dirs,
                probe_tokens=cfg.eval.trajectory_n_tokens,
                normalization_dir=normalization_dir,
                checkpoint_filename=checkpoint_filename,
                resume_checkpoint=resume_checkpoint,
            )

        run_streaming_training(
            sites=sites,
            token_stream=ResumableTokenStream(cfg.data, tokenizer),
            extractor=extractor,
            stream_cfg=cfg.stream,
            early_stopping=cfg.early_stopping,
            tokens_per_fwd=cfg.cache.tokens_per_fwd,
            seq_len=cfg.data.seq_len,
            device=cfg.stream.sae_device,
            out_root=out_root,
        )
    finally:
        extractor.close()
        for site in sites.values():
            site.close()


if __name__ == "__main__":
    main()
    # Triton can leave an autotuner worker thread alive after the CUDA work is
    # done, and normal interpreter shutdown then aborts in PyGILState_Release.
    # Every checkpoint and log is committed by now, so skip that shutdown path
    # (cli/cache_activations.py does the same).
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
