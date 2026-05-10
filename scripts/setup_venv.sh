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

# If a previous run left an incomplete .venv (no bin/activate), wipe and retry —
# this happens on Debian/Ubuntu when the matching pythonX.Y-venv package is
# missing: `python -m venv` exits with a partially-built directory.
if [ -d ".venv" ] && [ ! -f ".venv/bin/activate" ]; then
    echo ">>> Removing broken .venv (no bin/activate)"
    rm -rf .venv
fi

if [ ! -d ".venv" ]; then
    echo ">>> Creating .venv with $PY"
    if ! "$PY" -m venv .venv; then
        echo "FATAL: '$PY -m venv .venv' failed." >&2
        echo "On Debian/Ubuntu install the matching venv package, e.g.:" >&2
        echo "    apt install -y ${PY##*/}-venv python3-pip" >&2
        exit 4
    fi
fi

if [ ! -f ".venv/bin/activate" ]; then
    echo "FATAL: .venv/bin/activate missing after venv creation." >&2
    echo "Likely the ${PY##*/}-venv package is not installed. On Debian/Ubuntu:" >&2
    echo "    apt install -y ${PY##*/}-venv python3-pip" >&2
    echo "Then re-run this script (it will rebuild .venv)." >&2
    exit 4
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
