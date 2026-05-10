#!/usr/bin/env bash
# Runs train1.py in all 4 configurations sequentially:
#   1) scratch
#   2) continual, no reset
#   3) continual, reset high-freq branch
#   4) continual, reset low-freq branch
#
# Any extra args you pass to this script are forwarded to every run, e.g.:
#   ./scripts/run_all_modes.sh -seed 7 -nepochs 300 -optimizer adam

set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

EXTRA_ARGS=("$@")

run() {
    echo
    echo "=============================================================="
    echo ">>> python train1.py $*"
    echo "=============================================================="
    python train1.py "$@" "${EXTRA_ARGS[@]}"
}

run -training_mode scratch
run -training_mode continual -reset no
run -training_mode continual -reset high
run -training_mode continual -reset low

echo
echo "All 4 runs finished."
