#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
python "$ROOT/scripts/run_r16p14_stage2e_s0.py" --root "$ROOT"
python -m unittest discover -s "$ROOT/experiments/r16_p14_stage2e/tests" -v
python "$ROOT/scripts/verify_r16p14_stage2e_s0.py" --root "$ROOT"
