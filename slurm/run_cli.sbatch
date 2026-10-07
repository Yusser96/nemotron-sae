#!/usr/bin/env bash
# Run one sae_pipeline CLI module inside a container; arguments are passed through.
#
# Required environment:
#   REPO    checkout of this repository (visible inside the container)
#   IMAGE   container image with the needed dependencies (PyTorch + transformers for
#           hf_hooks, vLLM for vllm)
#   CLI     module name under sae_pipeline.cli, e.g. dump_activations or bench_extraction
# Optional environment:
#   EXTRA_PYTHONPATH  site-packages to put ahead of the image's own (e.g. a venv overlay)
#   PYTHON            interpreter inside the container (default: python3)
#   CONTAINER_MOUNTS  extra srun --container-mounts entries (comma separated)
#   HF_HOME           Hugging Face cache directory
#   GPU_SAMPLE_SECONDS  log GPU utilisation and memory every this many seconds
#
# Example:
#   sbatch --partition=<gpu-partition> --export=ALL,REPO=...,IMAGE=...,CLI=bench_extraction \
#       slurm/run_cli.sbatch --config configs/extraction_bench.yml --backend vllm
#SBATCH --job-name=sae-cli
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00

set -euo pipefail
: "${REPO:?Set REPO to the repository checkout}"
: "${IMAGE:?Set IMAGE to a container image}"
: "${CLI:?Set CLI to a sae_pipeline.cli module name}"
export EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-}" PYTHON="${PYTHON:-python3}" CLI GPU_SAMPLE_SECONDS="${GPU_SAMPLE_SECONDS:-}"
mounts="$REPO:$REPO${CONTAINER_MOUNTS:+,$CONTAINER_MOUNTS}"

srun --container-image="$IMAGE" --container-mounts="$mounts" --container-workdir="$REPO" bash -c '
set -euo pipefail
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [ -n "$GPU_SAMPLE_SECONDS" ]; then
    nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l "$GPU_SAMPLE_SECONDS" &
    trap "kill $!" EXIT
fi
PYTHONPATH="${EXTRA_PYTHONPATH:+$EXTRA_PYTHONPATH:}$PWD/src" "$PYTHON" -m "sae_pipeline.cli.$CLI" "$@"
' _ "$@"
