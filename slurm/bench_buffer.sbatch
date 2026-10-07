#!/usr/bin/env bash
# Measure directories as activation buffers with cli.bench_buffer: task 0 writes,
# task 1 reads, one directory after the other. With --nodes=2 the two tasks run on
# different nodes (a shared filesystem between a producer and a trainer job); with
# --nodes=1 they share a node. Arguments are passed to both roles.
#
# Required environment:
#   REPO         checkout of this repository (visible inside the container)
#   IMAGE        container image with PyTorch and safetensors
#   BUFFER_DIRS  space-separated parent directories; each run uses a fresh subdirectory
# Optional environment (as in run_cli.sbatch):
#   EXTRA_PYTHONPATH, PYTHON, CONTAINER_MOUNTS
#
# Example:
#   sbatch --nodes=2 --export=ALL,REPO=...,IMAGE=...,BUFFER_DIRS="/scratch/a /scratch/b" \
#       slurm/bench_buffer.sbatch --widths 2688 4096 --tokens 16384 --chunks 40
#SBATCH --job-name=sae-bench-buffer
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:30:00

set -euo pipefail
: "${REPO:?Set REPO to the repository checkout}"
: "${IMAGE:?Set IMAGE to a container image}"
: "${BUFFER_DIRS:?Set BUFFER_DIRS to the directories to measure}"
export EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-}" PYTHON="${PYTHON:-python3}"
mounts="$REPO:$REPO${CONTAINER_MOUNTS:+,$CONTAINER_MOUNTS}"

for parent in $BUFFER_DIRS; do
    dir="$parent/bench-buffer-$SLURM_JOB_ID"
    srun --container-image="$IMAGE" --container-mounts="$mounts" --container-workdir="$REPO" bash -c '
    set -euo pipefail
    role=$([ "$SLURM_PROCID" = 0 ] && echo write || echo read)
    PYTHONPATH="${EXTRA_PYTHONPATH:+$EXTRA_PYTHONPATH:}$PWD/src" \
        "$PYTHON" -m sae_pipeline.cli.bench_buffer --dir "$0" --role "$role" "$@"
    ' "$dir" "$@"
    rmdir "$dir"
done
