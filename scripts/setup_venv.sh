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

if [ "$(uname)" = "Darwin" ]; then
    echo ">>> macOS detected — using requirements-cpu.txt (skips CUDA-only packages)"
    python -m pip install -r "$REPO_ROOT/requirements-cpu.txt"
else
    # Linux / CUDA path. Two complications resolved here:
    #
    # 1. Default mirrors (incl. pypi-cache and pypi.ngc.nvidia.com) ship the
    #    cu124 torch wheel, which has no Blackwell (sm_100) kernels. On a B200
    #    that wheel "loads" but every CUDA op fails with "no kernel image
    #    available". The cu128 wheels on PyTorch's official index include
    #    sm_100 prebuilt — install torch from there explicitly.
    #
    # 2. mamba_ssm and causal-conv1d both `import torch` at the top of
    #    setup.py, but pip's PEP 517 build isolation hides torch from the
    #    build env, so they fail with "No module named 'torch'" even though
    #    torch is listed earlier. Fix is --no-build-isolation after torch is
    #    in place.
    #
    # 3. Their CUDA kernels need to be compiled with sm_100 in the arch list,
    #    so we set TORCH_CUDA_ARCH_LIST. nvcc ≥ 12.4 supports sm_100.
    echo ">>> Linux/CUDA — installing torch (cu128, includes Blackwell sm_100)"
    python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
        "torch>=2.7,<2.9"

    echo ">>> Installing build helpers (ninja, packaging)"
    python -m pip install ninja packaging

    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;9.0;10.0+PTX}"
    echo ">>> Building source extensions with TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

    echo ">>> Installing full requirements.txt with --no-build-isolation"
    python -m pip install --no-build-isolation -r "$REPO_ROOT/requirements.txt"

    # Build mamba_ssm + causal-conv1d from GitHub main against the current torch.
    # PyPI/NGC mamba_ssm 2.3.x has a c10::Warning constructor mismatch with
    # torch 2.8 (Blackwell-aware). Main branches carry the torch 2.8 fixes.
    #
    # IMPORTANT: --no-deps (not --force-reinstall). --force-reinstall treats
    # the declared `torch` dependency as a top-level requirement and re-pulls
    # whatever torch the user's pypi mirror serves (e.g. 2.11.0+cu130), which
    # then mismatches the system nvcc and fails compile. --no-deps preserves
    # the cu128 wheel we just installed.
    echo ">>> Pre-build sanity: which torch will mamba/causal-conv1d compile against?"
    python -c "import torch; print(f'torch={torch.__version__}  CUDA={torch.version.cuda}')"

    echo ">>> Removing any stale mamba_ssm / causal-conv1d binaries"
    python -m pip uninstall -y mamba_ssm causal-conv1d 2>/dev/null || true

    echo ">>> Building causal-conv1d from GitHub main (torch 2.8 ABI, --no-deps)"
    python -m pip install --no-build-isolation --no-deps \
        "git+https://github.com/Dao-AILab/causal-conv1d.git"

    echo ">>> Building mamba_ssm from GitHub main (torch 2.8 ABI, --no-deps)"
    python -m pip install --no-build-isolation --no-deps \
        "git+https://github.com/state-spaces/mamba.git"

    echo ">>> Post-build sanity: did torch survive intact?"
    python -c "import torch; print(f'torch={torch.__version__}  CUDA={torch.version.cuda}')"
fi

# Editable install of the project itself.
python -m pip install -e "$REPO_ROOT"

echo ">>> Done. Activate with: source $VENV/bin/activate"
