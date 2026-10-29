# nemotron-sae

Train and evaluate JumpReLU sparse autoencoders (SAEs) on the activations of Nemotron 3 Nano and similar hybrid Mamba–attention–MoE models. The repository covers training, and evaluation of reconstruction fidelity, behavioural fidelity (cross-entropy with the SAE patched in) and how many latents are actually used. Extracting activations at scale is delegated to [nemotron-extraction](https://github.com/alobo01/nemotron-extraction). Interpreting the trained features is left to downstream tools.

## Install

Python 3.10 or later.

```sh
python -m pip install -e '.[dev]'           # CPU: training on caches, tests, the toy below
python -m pip install -e '.[gpu,dev]'       # CUDA host: model inference as well
```

Hybrid models also need compatible `mamba-ssm` and `causal-conv1d` builds; `bash scripts/setup_venv.sh` installs them on a CUDA host. Access tokens are read from the environment variable named by `model.token_env_var`, never from configuration files.

## Try it on a CPU

[`examples/toy_sae.py`](examples/toy_sae.py) builds activations from a known sparse dictionary: each token mixes 3 of 64 random directions in 32 dimensions. It writes them as an activation cache, trains an SAE with the same trainer used for real models, and checks how much of the dictionary it recovered. It takes under a minute.

```sh
python examples/toy_sae.py outputs/toy_sae
```

```text
FVU 0.125, L0 3.00 (target 3), recovered 44/64 directions
```

FVU is the fraction of variance left unexplained, and L0 is the number of active latents per token. A direction counts as recovered when a decoder direction points the same way (cosine above 0.9).

## Train on a model

Every SAE belongs to one **site**: a layer and a component, named like `L26_resid_post`. See [hook sites](docs/hook_sites.md) for the model layout and the available components. Each command below takes a YAML configuration and has `--help`.

**From an activation cache.** This is the simplest path: one forward pass writes shards, then each SAE trains from disk.

```sh
python -m sae_pipeline.cli.cache_activations --config configs/dev.yaml --layer 25 --component resid_post
python -m sae_pipeline.cli.train_sae --config configs/dev.yaml --layer 25 --component resid_post --width 4096 --l0 30
```

**From a stream.** This path never stores the training activations, and trains many sites from the same forward passes:

```sh
python -m sae_pipeline.cli.train_sae_streaming --config configs/streaming.example.yml
```

To run the model and the trainers on different GPUs or nodes, start one producer, `produce_activations --out DIR --consumers NAME...`, and one trainer per consumer with `--fills DIR --consumer NAME`. Rerunning the same command resumes a stopped run from its last committed state.

**Evaluate and plot:**

```sh
python -m sae_pipeline.cli.evaluate --config configs/dev.yaml --layer 25 --component resid_post --width 4096 --l0 30 --delta-ce
python -m sae_pipeline.cli.plot --config configs/dev.yaml --layer 25 --component resid_post --width 4096 --l0 30
```

`--delta-ce` loads the model and measures how much next-token cross-entropy rises with the SAE in place. Without it, evaluation reads only the cache. Use `--partition validation --language en` for held-out data. [`configs/`](configs) holds annotated examples for fine-tuning a released SAE and for streaming.

## How it works

- **Objective.** The SAE encodes with a JumpReLU: a latent fires only above its learned threshold. It is trained on reconstruction error plus a quadratic penalty that keeps L0 near a target. The threshold learns through a straight-through estimator with a window set by `bandwidth`. Two optional terms address unused latents:
  - an inverse-Simpson penalty that spreads activation mass across latents;
  - an AuxK loss that trains dead latents to explain the residual error.

  The penalties ramp in over a configurable number of steps (`objective.py`, `jumprelu.py`).
- **Initialisation.** A run starts from random weights, from a released SAE (warm start, fresh optimiser) or from a partial reset of one.
- **Validation and selection.** Fixed held-out probes, per language, are scored at intervals. The cached trainer stops when FVU, explained variance and L0 are stable. The streaming trainer stops when the held-out reconstruction loss plateaus. Separately, it exports the checkpoint with the best full validation objective. That export can date from the penalty ramp, so compare it with the final export before choosing one.
- **Durability.** Checkpoints are weight and state pairs written atomically. Resuming restores weights, optimiser, random state, stream cursor and selection state. Raw exports fold normalisation into the weights for public release.

## Repository layout

| Path | Contents |
|---|---|
| `src/sae_pipeline/config.py` | Typed configuration for every stage |
| `src/sae_pipeline/model/` | Model loading and the topology map from site names to modules |
| `src/sae_pipeline/hooks/` | Component hooks, attention pre-projection capture, SAE patching for cross-entropy |
| `src/sae_pipeline/extraction/` | Extraction backends behind one interface (PyTorch hooks, vLLM, nemotron-extraction adapters), plus the producer/consumer fill directory and rolling activation pool |
| `src/sae_pipeline/data/` | Token streams from Hugging Face datasets |
| `src/sae_pipeline/cache/` | Activation shard writer, reader and manifest |
| `src/sae_pipeline/sae/` | The SAE, objective, cached and streaming trainers, initialisation, validation, stopping, checkpoints and resets |
| `src/sae_pipeline/eval/` | Reconstruction, utilisation and cross-entropy metrics; plots |
| `src/sae_pipeline/cli/` | Command-line entry points (cache, train, evaluate, plot, produce, compare, benchmarks) |
| `examples/` | The CPU toy above |
| `configs/` | Example configurations |
| `scripts/`, `slurm/` | Environment set-up and generic cluster launchers |
| `docs/hook_sites.md` | Model layout and site naming |
| `tests/` | CPU tests on synthetic activations; GPU tests are marked `cuda` |

## Development

```sh
python -m pytest -q -m 'not cuda' -n 6
python -m ruff check src tests examples
```

Tests of the nemotron-extraction adapters run only when that package is installed (`.[dev,extraction]`).
