#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
PYTHON_BIN="${PYTHON_BIN:-/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/libero_sim/bin/python}"
export PYTHONPATH="${REPO_ROOT}/experiments/r16_p14_stage2d:${REPO_ROOT}/experiments/r16_p14_stage2c:${REPO_ROOT}/experiments/r16_p14_stage2b:${REPO_ROOT}/experiments/r16_p14_stage2a:${REPO_ROOT}/experiments/r16_p14_libero_stage1:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

"${PYTHON_BIN}" -m r16_p14_stage2d.source_freeze
"${PYTHON_BIN}" -m r16_p14_stage2d.init_pool
"${PYTHON_BIN}" -m r16_p14_stage2d.events
"${PYTHON_BIN}" -m r16_p14_stage2d.isolation
"${PYTHON_BIN}" -m r16_p14_stage2d.qualification
"${PYTHON_BIN}" -m r16_p14_stage2d.calibration
"${PYTHON_BIN}" -m r16_p14_stage2d.freeze_rule
"${PYTHON_BIN}" -m r16_p14_stage2d.confirmatory
"${PYTHON_BIN}" -m r16_p14_stage2d.oracle_appendix
"${PYTHON_BIN}" -m r16_p14_stage2d.statistics
"${PYTHON_BIN}" -m r16_p14_stage2d.mechanism_reverse
"${PYTHON_BIN}" -m r16_p14_stage2d.report
