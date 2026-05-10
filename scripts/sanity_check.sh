#!/usr/bin/env bash
# Verify the runtime is actually usable: CUDA visible, mamba_ssm imports,
# causal-conv1d imports, transformers can fetch the model config (no full weight
# load — just the config metadata).
#
# Run AFTER setup_venv.sh has installed deps. Exits non-zero on the first
# failure so you can wire it into CI / shell `set -e` flows.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source "$REPO_ROOT/.venv/bin/activate"

echo ">>> torch + CUDA"
python - <<'PY'
import torch
print(f"torch        : {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda devices : {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  [{i}] {p.name}  total_mem={p.total_memory/2**30:.1f} GiB  cc={p.major}.{p.minor}")
else:
    raise SystemExit("FATAL: torch.cuda.is_available() == False")
PY

echo ">>> mamba_ssm"
python - <<'PY'
import mamba_ssm
print(f"mamba_ssm    : {mamba_ssm.__version__ if hasattr(mamba_ssm, '__version__') else 'imported (no version attr)'}")
PY

echo ">>> causal_conv1d"
python - <<'PY'
import causal_conv1d
print(f"causal_conv1d: {causal_conv1d.__version__ if hasattr(causal_conv1d, '__version__') else 'imported (no version attr)'}")
PY

echo ">>> transformers + model config (lightweight; no weights downloaded)"
python - <<'PY'
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained(
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    trust_remote_code=True,
)
print(f"model_type   : {cfg.model_type}")
print(f"hidden_size  : {getattr(cfg, 'hidden_size', '?')}")
print(f"num_layers   : {getattr(cfg, 'num_hidden_layers', getattr(cfg, 'num_layers', '?'))}")
print(f"vocab_size   : {getattr(cfg, 'vocab_size', '?')}")
PY

echo ">>> sae_pipeline imports"
python - <<'PY'
from sae_pipeline.config import PipelineCfg
from sae_pipeline.model.loader import load_model_and_tokenizer
from sae_pipeline.cache.writer import ShardWriter
from sae_pipeline.sae.jumprelu import JumpReLUSAE
from sae_pipeline.eval.plots import plot_training_curves
print("OK")
PY

echo ">>> sanity check passed"
