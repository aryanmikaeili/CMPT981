#!/usr/bin/env bash
# Run all 4 further_experiments sequentially, each in its OWN W&B project.
#
# Project naming
# --------------
#   ${PROJECT_PREFIX}-exp1-swap-pe       (input swap: which branch sees PE)
#   ${PROJECT_PREFIX}-exp2-symmetric     (symmetric-capacity ablation)
#   ${PROJECT_PREFIX}-exp3-frequency     (literal frequency-decomposed loss)
#   ${PROJECT_PREFIX}-exp4-flipped       (flipped non-stationarity dataset)
#
# Each project will end up containing 4 runs (scratch, continual_no,
# continual_<...>, continual_<...>) -- the four-mode comparison from
# the original train1.py, replicated under each experimental change.
#
# Usage
# -----
#   bash further_experiments/run_all_experiments.sh
#   bash further_experiments/run_all_experiments.sh -seed 7 -nepochs 300
#
# Override the W&B project prefix via env var:
#   PROJECT_PREFIX=my-prefix bash further_experiments/run_all_experiments.sh
#
# Caveats
# -------
# * Extra args after the script name are forwarded VERBATIM to all four
#   experiments. Use only flags that are valid in every experiment's
#   train.py (e.g. -seed, -nepochs, -lr, -image_size, -viz_every).
#   For experiment-specific flags (like exp2's -symmetric_width or
#   exp3's -blur_sigma) call that experiment's run_all.sh directly.
# * The exp4 dataset is auto-generated on first run; subsequent runs
#   reuse the on-disk images.
# * Make sure WANDB_API_KEY is set (in your shell env or in `.env` at
#   the repo root) before launching.

set -euo pipefail

cd "$(dirname "$0")"

PROJECT_PREFIX="${PROJECT_PREFIX:-cmpt981-plasticity}"
EXTRA_ARGS=("$@")

if [[ -f ../.venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source ../.venv/bin/activate
fi

start=$(date +%s)

echo
echo "=================================================================="
echo "Running all 4 further_experiments sequentially"
echo "Project prefix : ${PROJECT_PREFIX}"
echo "Extra args     : ${EXTRA_ARGS[*]:-(none)}"
echo "=================================================================="

bash exp1_swap_pe/run_all.sh \
    -project "${PROJECT_PREFIX}-exp1-swap-pe" "${EXTRA_ARGS[@]}"

bash exp2_symmetric_capacity/run_all.sh \
    -project "${PROJECT_PREFIX}-exp2-symmetric" "${EXTRA_ARGS[@]}"

bash exp3_frequency_decomposition/run_all.sh \
    -project "${PROJECT_PREFIX}-exp3-frequency" "${EXTRA_ARGS[@]}"

bash exp4_flipped_nonstationarity/run_all.sh \
    -project "${PROJECT_PREFIX}-exp4-flipped" "${EXTRA_ARGS[@]}"

end=$(date +%s)
elapsed=$((end - start))

echo
echo "=================================================================="
printf 'All 4 experiments finished in %dh %dm %ds.\n' \
    $((elapsed / 3600)) $(((elapsed % 3600) / 60)) $((elapsed % 60))
echo "=================================================================="
