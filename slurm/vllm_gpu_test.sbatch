#!/usr/bin/env bash
# Run the real-engine vLLM extraction tests inside a container that ships vLLM.
#
# Required environment:
#   REPO            checkout of this repository (visible inside the container)
#   IMAGE           container image with vLLM installed (e.g. an NGC vLLM image)
#   SAE_TEST_MODEL  Hugging Face id or local path of the model to extract from
# Optional environment:
#   CONTAINER_MOUNTS  extra srun --container-mounts entries (comma separated)
#   OVERLAY           directory for test-only packages installed with pip --target
#   SAE_TEST_SINK     directory for the hidden-state files (e.g. /dev/shm)
#   HF_HOME           Hugging Face cache directory
#
# Choose partition, GPU and time limit on the command line, e.g.
#   sbatch --partition=<gpu-partition> --export=ALL,REPO=...,IMAGE=...,SAE_TEST_MODEL=... \
#          slurm/vllm_gpu_test.sbatch
#SBATCH --job-name=vllm-gpu-test
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00

set -euo pipefail
: "${REPO:?Set REPO to the repository checkout}"
: "${IMAGE:?Set IMAGE to a container image with vLLM}"
: "${SAE_TEST_MODEL:?Set SAE_TEST_MODEL to the model to extract from}"
export SAE_TEST_MODEL SAE_TEST_SINK="${SAE_TEST_SINK:-}" HF_HOME="${HF_HOME:-}"
export OVERLAY="${OVERLAY:-$REPO/.overlay-vllm-tests}"
mounts="$REPO:$REPO${CONTAINER_MOUNTS:+,$CONTAINER_MOUNTS}"

srun --container-image="$IMAGE" --container-mounts="$mounts" --container-workdir="$REPO" bash -c '
set -euo pipefail
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h /dev/shm | tail -1
if ! PYTHONPATH="$OVERLAY" python3 -c "import pytest" 2>/dev/null; then
    # --no-deps: never shadow the packages the container was built with.
    python3 -m pip install --quiet --no-deps --target "$OVERLAY" pytest pluggy iniconfig
fi
[[ -z "$HF_HOME" ]] && unset HF_HOME
PYTHONPATH="$PWD/src:$OVERLAY" python3 -m pytest tests/test_vllm_extraction_gpu.py -m cuda -v -s -p no:cacheprovider
'
