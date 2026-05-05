#!/usr/bin/env bash
# Create .venv and install dependencies.
# On macOS we install the CPU subset (skips mamba_ssm + causal-conv1d, which need CUDA).
# On Linux + CUDA we install the full requirements.txt.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Pick the newest Python ≥ 3.10 we can find (project requires 3.10+).
if [ -z "${PY:-}" ]; then
    for cand in python3.12 python3.11 python3.10 python3; do
        if command -v "$cand" >/dev/null 2>&1; then
            PY="$cand"
            break
        fi
    done
fi
echo ">>> Using Python interpreter: $PY ($($PY --version))"

if [ ! -d ".venv" ]; then
    echo ">>> Creating .venv with $PY"
    "$PY" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip wheel setuptools

REQ_FILE="requirements.txt"
if [ "$(uname)" = "Darwin" ]; then
    echo ">>> macOS detected — using requirements-cpu.txt (skips CUDA-only packages)"
    REQ_FILE="requirements-cpu.txt"
fi

python -m pip install -r "$REQ_FILE"

# Editable install of the project itself.
python -m pip install -e .

echo ">>> Done. Activate with: source .venv/bin/activate"
