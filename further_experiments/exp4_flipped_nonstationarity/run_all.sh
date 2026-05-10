#!/usr/bin/env bash
# Runs exp4 (flipped non-stationarity: circles fixed, gradient varies)
# in all four configurations:
#   1) scratch
#   2) continual, no reset
#   3) continual, reset low-freq branch  (raw-xy input)
#   4) continual, reset high-freq branch (PE input)
#
# The dataset is auto-generated on the first run from `-data_seed`
# (default 42) into `data/circles_fixed_grad_varying/` -- 50 images,
# 256x256, identical fixed circles overlaid on a per-image gradient.
#
# Examples:
#   ./run_all.sh -project cmpt981-plasticity
#   ./run_all.sh -project cmpt981-plasticity -seed 7 -nepochs 300

set -euo pipefail

cd "$(dirname "$0")"

if [[ -f ../../.venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source ../../.venv/bin/activate
fi

EXTRA_ARGS=("$@")

run() {
    echo
    echo "=============================================================="
    echo ">>> python train.py $*"
    echo "=============================================================="
    python train.py "$@" "${EXTRA_ARGS[@]}"
}

run -training_mode scratch
run -training_mode continual -reset no
run -training_mode continual -reset low
run -training_mode continual -reset high

echo
echo "All 4 exp4 runs finished."
