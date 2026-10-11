#!/usr/bin/env bash
# End-to-end Stage-2 run on a fresh CUDA Linux box (for example an AWS g5/g6 instance).
#
#   scripts/stage2_gpu_run.sh [EXEMPLARS] [OUT]
#
# EXEMPLARS  intent exemplar file; default: core's built-in set in a sibling core checkout
#            (../core/crates/caliban-route/data/exemplars.default.json). Copy it over if the
#            box has no core checkout.
# OUT        output dir (default runs/stage2-eval, git-ignored).
#
# Environment: VENV (default .venv-train), TORCH_INDEX (default the PyTorch CUDA 13.0 index),
# STAGE2_BASES (default "modernbert mmbert"), HF_TOKEN (optional), MIN_FREE_MB (default 6000),
# QUANTIZATION (none | int8_dynamic, default none; fp32 artifacts still get an INT8 probe).
#
# HOME must be set: pip, torch and the Hugging Face libraries put caches under it. Hosts
# driven by AWS SSM Run Command start shells without HOME, so it is derived from the passwd
# entry here (and falls back to a directory inside the repo).
#
# Steps: install [train,export,dev] into a separate venv -> CPU smoke test (tiny random
# ModernBERT: train, export, ONNX parity; catches library API drift in minutes) -> fetch the
# pinned base models (SHA-256 checked) -> GPU smoke test on a real base model -> full run
# (scripts/router_stage2_eval.py: both backbones per base, kNN baselines, held-out intents,
# English paraphrases, French and Arabic, INT8 export + parity) -> strict manifest verify of
# every exported artifact (intent heads and the fine-tuned encoders as embedders).
# Results: $OUT/report.md and $OUT/summary.json.
set -euo pipefail

cd "$(dirname "$0")/.."
if [ -z "${HOME:-}" ]; then
  HOME="$(getent passwd "$(id -un)" | cut -d: -f6 || true)"
  [ -n "$HOME" ] && [ -w "$HOME" ] || HOME="$PWD/.home"
  mkdir -p "$HOME"
  export HOME
  echo "HOME was unset; using $HOME"
fi
EXEMPLARS="${1:-../core/crates/caliban-route/data/exemplars.default.json}"
OUT="${2:-runs/stage2-eval}"
VENV="${VENV:-.venv-train}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
STAGE2_BASES="${STAGE2_BASES:-modernbert mmbert}"
PY="$VENV/bin/python"

[ -f "$EXEMPLARS" ] || { echo "exemplar file not found: $EXEMPLARS" >&2; exit 1; }
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv

if [ ! -x "$PY" ]; then
  python3 -m venv "$VENV"
fi
"$PY" -m pip install -q --upgrade pip
"$PY" -m pip install -q torch --index-url "$TORCH_INDEX"
"$PY" -m pip install -q -e '.[train,export,dev]'
"$PY" -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('torch', torch.__version__, torch.cuda.get_device_name(0))"

echo "== CPU smoke test (tiny random ModernBERT)"
"$PY" -m pytest -q -m "integration and not gpu" tests/test_stage2_integration.py

echo "== fetch base models"
# shellcheck disable=SC2086
mapfile -t BASES < <("$PY" scripts/fetch_stage2_bases.py $STAGE2_BASES)

echo "== GPU smoke test (${BASES[0]%%=*})"
first="${BASES[0]#*=}"
IFS=: read -r base_dir base_id base_rev base_lic <<< "$first"
CALIBAN_ML_GPU_BASE_MODEL="$base_dir" CALIBAN_ML_GPU_BASE_ID="$base_id" \
  CALIBAN_ML_GPU_BASE_REVISION="$base_rev" CALIBAN_ML_GPU_BASE_LICENCE="$base_lic" \
  "$PY" -m pytest -q -m gpu tests/test_stage2_integration.py

echo "== full run"
args=()
for b in "${BASES[@]}"; do args+=(--base "$b"); done
"$PY" scripts/router_stage2_eval.py --exemplars "$EXEMPLARS" --out "$OUT" \
  --min-free-mb "${MIN_FREE_MB:-6000}" --quantization "${QUANTIZATION:-none}" "${args[@]}"

echo "== verify exported artifacts"
for d in "$OUT"/*/*/main/artifact "$OUT"/*/*/main/embedder; do
  [ -d "$d" ] || continue
  "$VENV/bin/caliban-ml" manifest verify --strict "$d"
done
echo "done: $OUT/report.md"
