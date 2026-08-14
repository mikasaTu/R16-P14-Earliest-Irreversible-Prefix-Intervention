#!/usr/bin/env bash
set -Eeuo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
experiment_root="$repo_root/experiments/r16_p14_stage2b"
artifact_root="$repo_root/artifacts/stage2b"
stage1_root="$repo_root/experiments/r16_p14_libero_stage1"
stage2a_root="$repo_root/experiments/r16_p14_stage2a"
r16p14_python=${R16P14_PYTHON:-/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/libero_sim/bin/python}

export PYTHONPATH="$experiment_root:$stage2a_root:$stage1_root${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}

case "${1:-help}" in
  freeze)
    "$r16p14_python" -m r16_p14_stage2b.freeze --output-dir "$artifact_root/source_freeze"
    ;;
  chunk-seed)
    seed=${2:?usage: commands.sh chunk-seed SEED [DEVICE] [TASK]}
    device=${3:-cuda}
    task=${4:-}
    task_args=()
    if [[ -n "$task" ]]; then
      task_args=(--task "$task")
    fi
    "$r16p14_python" -m r16_p14_stage2b.chunk_executability --seed "$seed" --device "$device" "${task_args[@]}"
    ;;
  chunk-aggregate)
    "$r16p14_python" -m r16_p14_stage2b.chunk_executability --aggregate
    ;;
  events-seed)
    seed=${2:?usage: commands.sh events-seed SEED [DEVICE]}
    device=${3:-cuda}
    "$r16p14_python" -m r16_p14_stage2b.actor_event_builder --seed "$seed" --device "$device"
    ;;
  events-aggregate)
    "$r16p14_python" -m r16_p14_stage2b.actor_event_builder --aggregate
    ;;
  perturbations)
    device=${2:-cuda}
    "$r16p14_python" -m r16_p14_stage2b.actor_perturbation_qualification --device "$device"
    ;;
  replay)
    device=${2:-cuda}
    "$r16p14_python" -m r16_p14_stage2b.actor_history_replay --device "$device"
    ;;
  checkpoint-report)
    "$r16p14_python" -m r16_p14_stage2b.checkpoint_report
    ;;
  atlas-smoke)
    device=${2:-cuda}
    "$r16p14_python" -m r16_p14_stage2b.atlas_runner --smoke --device "$device"
    ;;
  atlas-seed)
    seed=${2:?usage: commands.sh atlas-seed SEED [DEVICE]}
    device=${3:-cuda}
    "$r16p14_python" -m r16_p14_stage2b.atlas_runner --seed "$seed" --device "$device"
    ;;
  atlas-aggregate)
    "$r16p14_python" -m r16_p14_stage2b.atlas_aggregate
    ;;
  operator-seed)
    seed=${2:?usage: commands.sh operator-seed SEED [DEVICE]}
    device=${3:-cuda}
    "$r16p14_python" -m r16_p14_stage2b.operator_audit --seed "$seed" --device "$device"
    ;;
  operator-aggregate)
    "$r16p14_python" -m r16_p14_stage2b.operator_audit --aggregate
    ;;
  mechanism-audit)
    device=${2:-cuda}
    "$r16p14_python" -m r16_p14_stage2b.mechanism_audit --device "$device"
    ;;
  report)
    "$r16p14_python" -m r16_p14_stage2b.report
    ;;
  checksums)
    "$r16p14_python" -m r16_p14_stage2b.checksums
    ;;
  test)
    "$r16p14_python" -m r16_p14_stage2b.test_runner
    ;;
  help|*)
    printf '%s\n' 'usage: commands.sh {freeze|chunk-seed SEED [DEVICE] [TASK]|chunk-aggregate|events-seed SEED [DEVICE]|events-aggregate|perturbations [DEVICE]|replay [DEVICE]|checkpoint-report|atlas-smoke [DEVICE]|atlas-seed SEED [DEVICE]|atlas-aggregate|operator-seed SEED [DEVICE]|operator-aggregate|mechanism-audit [DEVICE]|report|checksums|test}'
    ;;
esac
