#!/usr/bin/env bash
# One-shot GPU deploy + dev smoke test.
#
# Run THIS SCRIPT ON A CUDA HOST. It:
#   1. Verifies CUDA (nvidia-smi + nvcc)
#   2. Clones / updates Yusser96/nemotron-sae
#   3. Materializes env.sh from inherited HF_TOKEN [+ OPENAI_API_KEY]
#   4. Sets up .venv (full requirements.txt → installs mamba_ssm + causal-conv1d;
#      can take 10–30 min the first time as those packages compile CUDA kernels)
#   5. Runs sanity_check.sh — proves the install works end-to-end
#   6. Dumps model_topology.json — confirms our regex classifier matches the real
#      trust_remote_code module names (catches naming-drift bugs cheap, before
#      we burn forward-pass time)
#   7. Runs dev_smoke_test.sh — cache 100 docs at (layer 25, resid_post),
#      train 1000-step JumpReLU SAE, evaluate, generate plots
#   8. Tars outputs/ into a timestamped artifact bundle
#
# Required env:
#   HF_TOKEN              — Hugging Face token (gated dataset access etc.)
#
# Optional env:
#   OPENAI_API_KEY        — for auto-interp eval (dev config has it disabled by default)
#   GH_TOKEN              — GitHub PAT for the private repo clone (alternatively, run
#                           `gh auth login` once on this host beforehand and the
#                           default git credential helper will pick it up)
#   REPO_URL              — override (default: https://github.com/Yusser96/nemotron-sae.git)
#   WORK_DIR              — override (default: $HOME/nemotron-sae)
#   BRANCH                — override (default: main)
#   SKIP_SETUP=1          — re-use an already-built .venv
#   SKIP_TOPOLOGY=1       — skip model topology dump (saves a model load if you
#                           know your runtime is fine)
#
# Exit codes:
#   0  success
#   2  no nvidia-smi (not a CUDA host)
#   3  no HF_TOKEN
#   *  any sub-step failed (set -e)

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/Yusser96/nemotron-sae.git}"
WORK_DIR="${WORK_DIR:-$HOME/nemotron-sae}"
BRANCH="${BRANCH:-main}"

# ----- 1. CUDA check ---------------------------------------------------------
echo "============================================================"
echo " Step 1/8: CUDA host check"
echo "============================================================"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "FATAL: nvidia-smi not found — this is not a CUDA host." >&2
    exit 2
fi
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv

if ! command -v nvcc >/dev/null 2>&1; then
    echo "WARN: nvcc not on PATH. mamba_ssm and causal-conv1d compile CUDA"
    echo "      kernels at install time and may fail without it. Try:"
    echo "        - 'module load cuda'   (cluster modules)"
    echo "        - 'apt install nvidia-cuda-toolkit'   (Debian/Ubuntu)"
    echo "        - or set CUDA_HOME to your CUDA install"
fi

if [ -z "${HF_TOKEN:-}" ]; then
    echo "FATAL: HF_TOKEN not set. Export it before running this script." >&2
    exit 3
fi

# ----- 2. Sync repo ----------------------------------------------------------
echo
echo "============================================================"
echo " Step 2/8: Sync repo into $WORK_DIR  (branch=$BRANCH)"
echo "============================================================"
clone_url="$REPO_URL"
if [ -n "${GH_TOKEN:-}" ]; then
    # Inject the PAT into the URL for non-interactive clones on a fresh box.
    # Strip any pre-existing https:// prefix and rebuild with credentials.
    no_scheme="${REPO_URL#https://}"
    clone_url="https://x-access-token:${GH_TOKEN}@${no_scheme}"
fi

if [ ! -d "$WORK_DIR/.git" ]; then
    git clone --branch "$BRANCH" "$clone_url" "$WORK_DIR"
else
    echo "Existing checkout found; fetching latest..."
    git -C "$WORK_DIR" fetch origin "$BRANCH"
    git -C "$WORK_DIR" checkout "$BRANCH"
    git -C "$WORK_DIR" pull --ff-only origin "$BRANCH"
fi
cd "$WORK_DIR"

# ----- 3. env.sh -------------------------------------------------------------
echo
echo "============================================================"
echo " Step 3/8: Materialize env.sh"
echo "============================================================"
{
    echo "export HF_TOKEN='$HF_TOKEN'"
    if [ -n "${OPENAI_API_KEY:-}" ]; then
        echo "export OPENAI_API_KEY='$OPENAI_API_KEY'"
    fi
} > env.sh
chmod 600 env.sh
echo "Wrote $(pwd)/env.sh ($(wc -c < env.sh) bytes; mode 600)"

# ----- 4. Venv + deps --------------------------------------------------------
echo
echo "============================================================"
echo " Step 4/8: Set up .venv  (installs mamba_ssm, causal-conv1d, etc.)"
echo "============================================================"
if [ "${SKIP_SETUP:-0}" = "1" ] && [ -d ".venv" ]; then
    echo "SKIP_SETUP=1 and .venv exists — skipping."
else
    bash scripts/setup_venv.sh
fi

# shellcheck disable=SC1091
source "$WORK_DIR/.venv/bin/activate"
# shellcheck disable=SC1091
source "$WORK_DIR/env.sh"

# ----- 5. Sanity check -------------------------------------------------------
echo
echo "============================================================"
echo " Step 5/8: Runtime sanity check"
echo "============================================================"
bash scripts/sanity_check.sh

# ----- 6. Model topology -----------------------------------------------------
echo
echo "============================================================"
echo " Step 6/8: Dump model topology  (catches module-naming drift early)"
echo "============================================================"
if [ "${SKIP_TOPOLOGY:-0}" = "1" ]; then
    echo "SKIP_TOPOLOGY=1 — skipping."
else
    python -m sae_pipeline.model.topology \
        --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
        --print
fi

# ----- 7. Smoke test ---------------------------------------------------------
echo
echo "============================================================"
echo " Step 7/8: Dev smoke test  (cache → train → evaluate → plot)"
echo "============================================================"
bash scripts/dev_smoke_test.sh

# ----- 8. Pack artifacts -----------------------------------------------------
echo
echo "============================================================"
echo " Step 8/8: Pack artifacts"
echo "============================================================"
ts="$(date -u +%Y%m%dT%H%M%SZ)"
artifact="smoke_artifacts_${ts}.tar.gz"
tar -czf "$artifact" \
    outputs/caches/dev_smoke/*/manifest.json \
    outputs/checkpoints/dev_smoke/ \
    outputs/logs/dev_smoke/ \
    2>/dev/null || true

if [ -f "$artifact" ]; then
    ls -lh "$artifact"
    echo
    echo "✓ Smoke test complete."
    echo "  Artifact:   $WORK_DIR/$artifact"
    echo "  Plots:      $WORK_DIR/outputs/checkpoints/dev_smoke/L25_resid_post/jumprelu_w4096_l0_30/plots/"
    echo "             $WORK_DIR/outputs/logs/dev_smoke/L25_resid_post/jumprelu_w4096_l0_30_plots/"
    echo
    echo "  rsync the tarball back with e.g.:"
    echo "    rsync -av <gpu-host>:$WORK_DIR/$artifact ./"
else
    echo "WARN: no artifact tarball produced — check the logs above for failures."
fi
