#!/usr/bin/env bash
# Run the FlashDLM (guided diffusion) baseline on GSM8K-512 and print the
# accuracy / throughput / tokens-per-step summary.
#
#   bash scripts/run_gsm8k_512_benchmark.sh [config.yaml]
#
# Pins a single GPU so the measured throughput is a single-device number, and
# keeps wandb offline so the run does not block on a login prompt.
set -euo pipefail

CONFIG="${1:-test_configs/dream/gsm8k/guided_diffusion/gsm8k_512_5shot_paper_protocol.yaml}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM=false

echo "config : ${CONFIG}"
echo "gpu    : ${CUDA_VISIBLE_DEVICES}"

python guided_diffusion/dream_eval/gsm8k_guided_evaluator.py --config "${CONFIG}"

RUN_DIR="$(python - "${CONFIG}" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))["save"]["run_dir"])
PY
)"

echo
echo "=== aggregate metrics ==="
find "${RUN_DIR}" -name eval_results.json -exec \
    python scripts/summarize_flashdlm_metrics.py {} +
