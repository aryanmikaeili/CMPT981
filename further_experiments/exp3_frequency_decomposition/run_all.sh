#!/usr/bin/env bash
# Runs exp3 (literal frequency decomposition via blurred-GT/residual-GT
# auxiliary losses) in all four configurations:
#   1) scratch
#   2) continual, no reset
#   3) continual, reset low-freq branch  (now LITERALLY low-frequency)
#   4) continual, reset high-freq branch (now LITERALLY high-frequency)
#
# All extra args are forwarded to every run.  Use `-blur_sigma` to sweep
# how much of the signal is routed to the high branch.
#
# Examples:
#   ./run_all.sh -project cmpt981-plasticity                        # default sigma=4
#   ./run_all.sh -project cmpt981-plasticity -blur_sigma 2          # sharper low band
#   ./run_all.sh -project cmpt981-plasticity -blur_sigma 8          # blurrier low band

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
echo "All 4 exp3 runs finished."
