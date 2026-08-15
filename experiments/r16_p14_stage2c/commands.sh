#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/libero_sim/bin/python}
export PYTHONPATH="$PROJECT_ROOT/experiments/r16_p14_stage2c:$PROJECT_ROOT/experiments/r16_p14_stage2b:$PROJECT_ROOT/experiments/r16_p14_stage2a:$PROJECT_ROOT/experiments/r16_p14_libero_stage1${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

case "${1:-help}" in
  test)
    exec "$PYTHON_BIN" -m pytest -q "$PROJECT_ROOT/experiments/r16_p14_stage2c/tests"
    ;;
  freeze)
    exec "$PYTHON_BIN" -m r16_p14_stage2c.freeze
    ;;
  contract-audit)
    exec "$PYTHON_BIN" -m r16_p14_stage2c.contract_audit "${@:2}"
    ;;
  events)
    exec "$PYTHON_BIN" -m r16_p14_stage2c.events "${@:2}"
    ;;
  qualify)
    exec "$PYTHON_BIN" -m r16_p14_stage2c.qualification "${@:2}"
    ;;
  evaluate)
    exec "$PYTHON_BIN" -m r16_p14_stage2c.evaluation "${@:2}"
    ;;
  aggregate)
    exec "$PYTHON_BIN" -m r16_p14_stage2c.aggregate "${@:2}"
    ;;
  report)
    exec "$PYTHON_BIN" -m r16_p14_stage2c.report
    ;;
  *)
    echo "usage: $0 {test|freeze|contract-audit|events|qualify|evaluate|aggregate|report} [args...]" >&2
    exit 2
    ;;
esac

