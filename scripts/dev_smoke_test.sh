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
if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    source "$REPO_ROOT/.venv/bin/activate"
elif ! command -v python >/dev/null 2>&1; then
    echo 'Activate a Python environment or create .venv before running this script.' >&2
    exit 1
fi
# Optional: export the token variable named by model.token_env_var.
# shellcheck disable=SC1091
if [ -f "$REPO_ROOT/env.sh" ]; then source "$REPO_ROOT/env.sh"; fi

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
python - "$CONFIG" "$LAYER" "$COMPONENT" "$ARCH" "$WIDTH" "$L0" <<'PY'
import sys
from pathlib import Path
from sae_pipeline.config import PipelineCfg

config, layer, component, arch, width, l0 = sys.argv[1:]
cfg = PipelineCfg.from_yaml(config)
slug, stem = f"L{layer}_{component}", f"{arch}_w{width}_l0_{l0}"
paths = [
    Path(cfg.cache.cache_dir) / cfg.run_id / slug / "manifest.json",
    Path(cfg.sae.ckpt_dir) / cfg.run_id / slug / stem,
    Path(cfg.log.log_dir) / cfg.run_id / slug / "train" / f"{stem}_eval.json",
]
for path in paths:
    if not path.exists():
        raise SystemExit(f"Missing smoke-test artifact: {path}")
    print(path)
PY
