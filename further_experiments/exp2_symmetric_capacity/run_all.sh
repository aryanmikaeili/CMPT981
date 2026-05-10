#!/usr/bin/env bash
# Runs exp2 (symmetric-capacity ablation) in all four configurations:
#   1) scratch
#   2) continual, no reset
#   3) continual, reset low-freq branch (raw-xy input)
#   4) continual, reset high-freq branch (PE input)
#
# All extra args are forwarded to every run.  The W&B project name is
# required by train.py.  Use `-symmetric_width` / `-symmetric_layers`
# to sweep architectures while keeping the four-mode comparison fair.
#
# Examples:
#   ./run_all.sh -project cmpt981-plasticity                          # default 256/3
#   ./run_all.sh -project cmpt981-plasticity \
#                -symmetric_width 128 -symmetric_layers 2             # match small
#   ./run_all.sh -project cmpt981-plasticity \
#                -symmetric_width 192 -symmetric_layers 3             # ~original total

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
echo "All 4 exp2 runs finished."
