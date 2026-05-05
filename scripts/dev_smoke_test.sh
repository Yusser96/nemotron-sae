#!/usr/bin/env bash
# End-to-end dev pipeline: cache → train → eval on layer 25, component resid_post.
# Requires a CUDA host with the model weights downloaded (mamba_ssm + causal-conv1d).
# On macOS / CPU-only hosts this script will fail at the model-load step; that is
# expected — run `pytest tests/` instead, which exercises the same code paths
# end-to-end on synthetic activations without loading the LM.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source .venv/bin/activate
# shellcheck disable=SC1091
source env.sh   # HF_TOKEN, OPENAI_API_KEY

CONFIG="${CONFIG:-configs/dev.yaml}"
LAYER="${LAYER:-25}"
COMPONENT="${COMPONENT:-resid_post}"
ARCH="${ARCH:-jumprelu}"
WIDTH="${WIDTH:-4096}"
L0="${L0:-30}"

echo ">>> Phase A: cache activations  ($CONFIG L=$LAYER C=$COMPONENT)"
python -m sae_pipeline.cli.cache_activations \
    --config "$CONFIG" --layer "$LAYER" --component "$COMPONENT"

echo ">>> Phase B: train SAE  (arch=$ARCH width=$WIDTH L0=$L0)"
python -m sae_pipeline.cli.train_sae \
    --config "$CONFIG" --layer "$LAYER" --component "$COMPONENT" \
    --arch "$ARCH" --width "$WIDTH" --l0 "$L0"

echo ">>> Phase C: evaluate"
python -m sae_pipeline.cli.evaluate \
    --config "$CONFIG" --layer "$LAYER" --component "$COMPONENT" \
    --arch "$ARCH" --width "$WIDTH" --l0 "$L0"

echo ">>> Done. Artifacts:"
ls -la outputs/caches/dev_smoke/L${LAYER:?}_${COMPONENT}/manifest.json
ls -la outputs/checkpoints/dev_smoke/L${LAYER:?}_${COMPONENT}/${ARCH}_w${WIDTH}_l0_${L0}/
ls -la outputs/logs/dev_smoke/L${LAYER:?}_${COMPONENT}/
