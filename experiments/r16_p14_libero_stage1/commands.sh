#!/usr/bin/env bash
set -Eeuo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
experiment_root="$repo_root/experiments/r16_p14_libero_stage1"
r16p14_python=${R16P14_PYTHON:-python}

usage() {
  printf 'usage: %s {verify|test}\n' "$0"
}

case "${1:-}" in
  verify)
    "$r16p14_python" "$repo_root/scripts/verify_r16p14_release.py"
    ;;
  test)
    export PYTHONPATH="$repo_root:$experiment_root${PYTHONPATH:+:$PYTHONPATH}"
    export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-$experiment_root/libero_config}
    export MUJOCO_GL=${MUJOCO_GL:-egl}
    export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
    "$r16p14_python" -m pytest "$experiment_root/tests" -q
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
