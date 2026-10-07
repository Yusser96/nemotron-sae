#!/usr/bin/env bash
# Streaming SAE training with the model in its own vLLM process on the same node.
# cli.serve_activations runs vLLM on the first GPU and sends the activations GPU to
# GPU to cli.train_sae_streaming, which trains on stream.sae_device. Both read the
# same config, whose extraction.backend is vllm_remote. One task sees both GPUs so
# NCCL can copy peer to peer; the job ends with the trainer's exit status.
#
# Required environment:
#   REPO    checkout of this repository (visible inside the container)
#   IMAGE   container image with vLLM and the training dependencies
#   CONFIG  pipeline config, relative to REPO or absolute
# Optional environment (as in run_cli.sbatch):
#   EXTRA_PYTHONPATH, PYTHON, CONTAINER_MOUNTS, HF_HOME
#
# Example:
#   sbatch --partition=<gpu-partition> --export=ALL,REPO=...,IMAGE=...,CONFIG=... \
#       slurm/stream_train_remote.sbatch
#SBATCH --job-name=sae-stream-remote
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00

set -euo pipefail
: "${REPO:?Set REPO to the repository checkout}"
: "${IMAGE:?Set IMAGE to a container image}"
: "${CONFIG:?Set CONFIG to a pipeline config}"
export EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-}" PYTHON="${PYTHON:-python3}" CONFIG
mounts="$REPO:$REPO${CONTAINER_MOUNTS:+,$CONTAINER_MOUNTS}"

srun --container-image="$IMAGE" --container-mounts="$mounts" --container-workdir="$REPO" bash -c '
set -uo pipefail
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
nvidia-smi topo -m
export PYTHONPATH="${EXTRA_PYTHONPATH:+$EXTRA_PYTHONPATH:}$PWD/src"
"$PYTHON" -m sae_pipeline.cli.serve_activations --config "$CONFIG" &
producer=$!
"$PYTHON" -m sae_pipeline.cli.train_sae_streaming --config "$CONFIG" &
trainer=$!
wait -n -p finished "$producer" "$trainer"
status=$?
if [ "$finished" = "$producer" ]; then
    # The producer exits cleanly only once the trainer has told it to stop.
    if [ "$status" -ne 0 ]; then
        echo "The producer failed (exit $status)" >&2
        kill "$trainer"
    fi
    wait "$trainer"
    exit $?
fi
kill "$producer" 2>/dev/null
wait "$producer"
exit "$status"
'
