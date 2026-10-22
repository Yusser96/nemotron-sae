# nemotron-sae

Train and evaluate JumpReLU sparse autoencoders on model activations. The encoder
is tied to the decoder at initialization and optimized independently afterward.

This repository owns SAE training and evaluation of feature utilisation,
reconstruction fidelity and behavioural fidelity. Reusable activation-extraction
strategies belong to `nemotron-extraction`. Downstream autointerpretation,
multilingual feature exploration and the Web UI belong to a downstream
interpretation repository, which consumes the trained and evaluated SAEs.

## Install

Use Python 3.10 or newer. On a CUDA host, `bash scripts/setup_venv.sh` creates the
project environment and installs the CUDA dependencies. In an existing environment:

```sh
python -m pip install -e '.[gpu,dev]'
```

CPU development uses `python -m pip install -e '.[dev]'`. Hybrid model inference
also needs the model's compatible Mamba and causal convolution builds; see the
setup script. Credentials are read from the environment, never from configuration
values. Set `model.token_env_var` to the name of the environment variable to use.

## Train from cached activations

```sh
python -m sae_pipeline.cli.cache_activations --config configs/dev.yaml --layer 25 --component resid_post
python -m sae_pipeline.cli.train_sae --config configs/dev.yaml --layer 25 --component resid_post --width 4096 --l0 30
```

For exact model paths, configure `target.sites` and select a `--site` slug instead
of `--layer` and `--component`. `cache_activations --all-targets` captures all
configured sites in one forward pass. Build per-language probes with
`--partition validation --language en` (and each additional configured language).

## Train from a stream

Set `stream.enabled: true`, configure `target.sites`, and prepare complete bilingual
or monolingual validation caches under `validation_run_id` before training:

```sh
python -m sae_pipeline.cli.train_sae_streaming --config configs/streaming.example.yml
```

`sae.checkpoint_source` with `sae.initialization_mode: checkpoint` warm-starts a
released dictionary with a fresh optimizer. `initialization_mode: random` uses
fresh decoder-derived encoder weights. Configure fixed-LR plateau stopping with
`early_stopping.mode: objective`, `min_lr_scale: 1.0`, and a suitable minimum step
count and patience. Numerical safeguards remain active independently of plateau
stopping. Rerun the identical command to resume the committed stream cursor,
optimizer, random state and selection state after a time-limited pause.

To train from a separate activation producer, pass `--fills DIR --consumer NAME`,
with a distinct name for each trainer.

## Evaluate and plot

```sh
python -m sae_pipeline.cli.evaluate --config configs/dev.yaml --layer 25 --component resid_post --width 4096 --l0 30
python -m sae_pipeline.cli.plot --config configs/dev.yaml --layer 25 --component resid_post --width 4096 --l0 30
```

Evaluation reports reconstruction, firing sparsity and dictionary utilization.
Use `--partition validation --language en` for held-out measurements, and add
`--delta-ce` to measure next-token cross-entropy degradation with the SAE patched
into the model. That flag loads the language model and uses the configured
`eval.delta_ce_n_seqs` and `eval.delta_ce_seq_len`; it processes one sequence at a
time to bound logits memory. Reconstruction-only evaluation does not load it.

See the shipped [configuration examples](configs/finetuning.example.yml),
[streaming example](configs/streaming.example.yml) and [hook sites](docs/hook_sites.md).
Every CLI also supports `--help`. Installed aliases are `sae-cache`, `sae-train`,
`sae-train-streaming`, `sae-evaluate`, `sae-plot` and `sae-launch`.

## Cache and checkpoint compatibility

Cache manifests must match model, site, width, partition and language. Keep
normalization and centering metadata alongside canonical checkpoints. Public raw
exports already fold these transformations into the weights; select
`--checkpoint-format raw_export` when evaluating them. Streaming resume requires
the committed weight/state pair referenced by its stream-state file. Preserve the
pair and cursor together; a raw export alone cannot resume an optimizer.

`extraction.backend` defaults to `hf_hooks`; `vllm` and `vllm_remote` are also
built in. The optional `extraction` extra pins `nemotron-extraction==1.1.0` and
adds the `nemotron_extraction_hf` and `nemotron_extraction_vllm` backends. Check
them with `python -m sae_pipeline.cli.compare_activations` on your model and sites
before switching an existing experiment.

## Development

```sh
python -m pytest -q -m 'not cuda' -p no:cacheprovider -n 6
python -m ruff check --no-cache --select F src tests
```

The adapter tests skip unless `nemotron-extraction` is installed (`.[dev,extraction]`).

### Independent activation producers and SAE consumers

The maintained vLLM extractor can publish version-2 per-site receipts with device finite validation and parallel file writers. `sae_pipeline.extraction.publication.entry_from_receipt` registers those receipts without reading activation payloads again. The rolling reader accepts both per-site and legacy monolithic caches, verifies tensor checksums during prefetch, and reclaims data only after durable consumer commits. Missing shared metadata is retried for up to five seconds; persistent failures preserve state and stop the affected run.
