#!/usr/bin/env bash
# Stage 0 baseline reproduction: run the ungated model multiple times.
# Usage:
#   scripts/run_baseline.sh               # run 3 seeds: 1337,1338,1339 on 8 GPUs
#   NPROC=4 COUNT=5 BASE_SEED=2000 scripts/run_baseline.sh
# Env vars:
#   NPROC       - number of GPUs per node (default 8)
#   COUNT       - how many runs (default 3)
#   BASE_SEED   - first seed; seeds increment by 1 per run (default 1337)
#   SCRIPT      - training script path (default train_gpt.py)
# Notes:
#   - Gating is forced off (ATTNGATE=none).
#   - Requires torchrun and the dataset at data/fineweb10B/fineweb_*.
#   - Does not modify code; only launches runs.

set -euo pipefail

NPROC="${NPROC:-8}"
COUNT="${COUNT:-3}"
BASE_SEED="${BASE_SEED:-1337}"
SCRIPT="${SCRIPT:-train_gpt.py}"

for i in $(seq 0 $((COUNT-1))); do
  SEED=$((BASE_SEED + i))
  echo "==> Baseline run $((i+1))/$COUNT (seed=${SEED}, nproc=${NPROC})"
  ATTNGATE=none GATEPOS=sdpa GATEACT=sigmoid \
  SEED="${SEED}" \
  torchrun --standalone --nproc_per_node="${NPROC}" "${SCRIPT}"
done
