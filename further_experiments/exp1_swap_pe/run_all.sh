#!/usr/bin/env bash
# Runs exp1 (swap PE/raw between primary and residual branches) in all
# four configurations sequentially:
#   1) scratch
#   2) continual, no reset
#   3) continual, reset primary  (large, PE-input branch)
#   4) continual, reset residual (small, raw-xy branch)
#
# All extra args are forwarded to every run.  -project is required by
# train.py and is the canonical place to plug in the W&B project name:
#
#   ./run_all.sh -project cmpt981-plasticity
#   ./run_all.sh -project cmpt981-plasticity -seed 7 -nepochs 300

set -euo pipefail

cd "$(dirname "$0")"

# Activate the project venv if present (mirrors scripts/run_all_modes.sh).
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
run -training_mode continual -reset primary
run -training_mode continual -reset residual

echo
echo "All 4 exp1 runs finished."
