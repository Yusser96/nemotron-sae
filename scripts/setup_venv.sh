#!/usr/bin/env bash
# Create THE project venv at $REPO_ROOT/.venv and install dependencies.
# Always uses the same .venv at the repo root — never creates per-subdirectory
# venvs, regardless of where you invoke the script from.
#
# On macOS the CPU subset is installed (skips mamba_ssm + causal-conv1d).
# On Linux + CUDA the full requirements.txt is installed.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV="$REPO_ROOT/.venv"

# Quick guard: warn (and remove) if any stray .venv was created in a subdir.
# Search top-level once; deeper searches would be too slow on big trees.
for stray in $(find "$REPO_ROOT" -maxdepth 3 -type d -name ".venv" 2>/dev/null); do
    if [ "$stray" != "$VENV" ]; then
        echo "WARN: removing stray venv $stray (only $VENV is canonical)"
        rm -rf "$stray"
    fi
done

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

# If a previous run left an incomplete .venv (no bin/activate), wipe and retry.
if [ -d "$VENV" ] && [ ! -f "$VENV/bin/activate" ]; then
    echo ">>> Removing broken venv at $VENV (no bin/activate)"
    rm -rf "$VENV"
fi

if [ ! -d "$VENV" ]; then
    # Use the PyPI `virtualenv` package rather than stdlib `python -m venv` —
    # it bundles its own pip bootstrap, so it works on minimal containers
    # missing the `pythonX.Y-venv` apt package (e.g. the user's B200 host).
    if ! "$PY" -m virtualenv --version >/dev/null 2>&1; then
        echo ">>> Installing virtualenv via $PY -m pip"
        # Try in order: --user (works on PEP 668 systems like Homebrew Python),
        # plain (works on root containers / unrestricted Pythons), and finally
        # --break-system-packages as a last resort.
        "$PY" -m pip install --user virtualenv 2>/dev/null \
            || "$PY" -m pip install virtualenv 2>/dev/null \
            || "$PY" -m pip install --break-system-packages virtualenv
    fi
    echo ">>> Creating $VENV with virtualenv (--python=$PY)"
    "$PY" -m virtualenv --python="$PY" "$VENV"
fi

if [ ! -f "$VENV/bin/activate" ]; then
    echo "FATAL: $VENV/bin/activate missing after virtualenv creation." >&2
    echo "Verify pip + virtualenv work for $PY:" >&2
    echo "    $PY -m pip install virtualenv && $PY -m virtualenv $VENV" >&2
    exit 4
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

python -m pip install --upgrade pip wheel setuptools

REQ_FILE="$REPO_ROOT/requirements.txt"
if [ "$(uname)" = "Darwin" ]; then
    echo ">>> macOS detected — using requirements-cpu.txt (skips CUDA-only packages)"
    REQ_FILE="$REPO_ROOT/requirements-cpu.txt"
fi

python -m pip install -r "$REQ_FILE"

# Editable install of the project itself.
python -m pip install -e "$REPO_ROOT"

echo ">>> Done. Activate with: source $VENV/bin/activate"
