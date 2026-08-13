#!/usr/bin/env bash
set -Eeuo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
experiment_root="$repo_root/experiments/r16_p14_stage2a"
stage1_root="$repo_root/experiments/r16_p14_libero_stage1"
stage1b_root="$repo_root/experiments/r16_p14_libero_stage1b"
artifact_root="$repo_root/artifacts/stage2a"
r16p14_python=${R16P14_PYTHON:-/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/libero_sim/bin/python}
libero_config="$stage1_root/libero_config"
cache_root=/mnt/cpfs/zbl-cpfs-new/USERS/leon/cache/r16_p14_stage2a
checkpoint_root=/mnt/cpfs/zbl-cpfs-new/CKPT/leon/torch/r16_p14_stage2a
log_root=/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r16_p14_stage2a
run_id=r16p14-stage2a-act-v1

export PYTHONPATH="$repo_root:$stage1_root:$stage1b_root:$experiment_root${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}

case "${1:-help}" in
  freeze)
    "$r16p14_python" -m r16_p14_stage2a.freeze \
      --output-dir "$artifact_root/source_freeze"
    ;;
  clue)
    "$r16p14_python" -m r16_p14_stage2a.clue \
      --source "$repo_root/artifacts/stage1b/expert_chunk_calibration/selected_config_paired_metrics.csv" \
      --output-dir "$experiment_root/reports"
    ;;
  task-screen)
    "$r16p14_python" -m r16_p14_stage2a.task_screen \
      --config-dir "$libero_config" \
      --output-dir "$experiment_root/task_screen"
    ;;
  perturbations)
    "$r16p14_python" -m r16_p14_stage2a.perturbation_qualification \
      --config-dir "$libero_config" \
      --output-dir "$artifact_root/perturbations"
    ;;
  actor-cache)
    shift
    "$r16p14_python" -m r16_p14_stage2a.actor_data \
      --cache-root "$cache_root" --config-dir "$libero_config" "$@"
    ;;
  actor-smoke)
    smoke_cache=/mnt/cpfs/zbl-cpfs-new/USERS/leon/cache/r16_p14_stage2a_smoke
    "$r16p14_python" -m r16_p14_stage2a.actor_data \
      --tasks put_the_cream_cheese_in_the_bowl open_the_middle_drawer_of_the_cabinet \
      --cache-root "$smoke_cache" --config-dir "$libero_config" \
      --demo-count 2 --max-states-per-demo 32
    CUDA_VISIBLE_DEVICES=${R16P14_SMOKE_GPU:-0} "$r16p14_python" -m r16_p14_stage2a.actor_train \
      --tasks put_the_cream_cheese_in_the_bowl open_the_middle_drawer_of_the_cabinet \
      --seed 7 --run-id r16p14-stage2a-act-smoke-v1 --cache-root "$smoke_cache" \
      --checkpoint-root "$checkpoint_root/smoke" --log-root "$log_root/smoke" \
      --artifact-root "$artifact_root/actor/smoke" --steps 3 --batch-size 4 \
      --save-interval 3 --device cuda
    ;;
  actor-train)
    seed=${2:?usage: commands.sh actor-train SEED [DEVICE]}
    device=${3:-cuda}
    "$r16p14_python" -m r16_p14_stage2a.actor_train \
      --seed "$seed" --run-id "$run_id" --cache-root "$cache_root" \
      --checkpoint-root "$checkpoint_root" --log-root "$log_root" \
      --artifact-root "$artifact_root/actor" --device "$device"
    ;;
  actor-eval)
    seed=${2:?usage: commands.sh actor-eval SEED [DEVICE]}
    device=${3:-cuda}
    "$r16p14_python" -m r16_p14_stage2a.actor_eval \
      --seed "$seed" --checkpoint "$artifact_root/actor/checkpoints/seed_${seed}.pt" \
      --output-dir "$artifact_root/actor/eval" --config-dir "$libero_config" \
      --device "$device"
    ;;
  actor-aggregate)
    "$r16p14_python" -m r16_p14_stage2a.actor_aggregate \
      --input-dir "$artifact_root/actor/eval" \
      --output-dir "$artifact_root/actor"
    ;;
  actor-mechanism-audit)
    CUDA_VISIBLE_DEVICES=${R16P14_AUDIT_GPU:-0} "$r16p14_python" -m r16_p14_stage2a.actor_mechanism_audit \
      --checkpoint-dir "$artifact_root/actor/checkpoints" \
      --qualification-summary "$artifact_root/actor/qualification_summary.json" \
      --weak-summary "$repo_root/artifacts/formal_pilot/report/experiment_summary.json" \
      --output-dir "$artifact_root/actor/mechanism_audit" \
      --cache-root "$cache_root" --device cuda
    ;;
  atlas-preflight)
    "$r16p14_python" -m r16_p14_stage2a.atlas_preflight \
      --actor-summary "$artifact_root/actor/qualification_summary.json" \
      --frozen-perturbations "$artifact_root/perturbations/frozen_parameters.json" \
      --grid-summary "$artifact_root/perturbations/grid_summary.json" \
      --replay-summary "$artifact_root/replay/summary.json" \
      --output-dir "$artifact_root/atlas"
    ;;
  checkpoint-report)
    "$r16p14_python" -m r16_p14_stage2a.checkpoint_report \
      --repo "$repo_root" --output-dir "$experiment_root/reports"
    ;;
  replay)
    "$r16p14_python" -m r16_p14_stage2a.replay_gate \
      --config-dir "$libero_config" \
      --frozen-perturbations "$artifact_root/perturbations/frozen_parameters.json" \
      --task-screen "$experiment_root/task_screen/task_roster_frozen.json" \
      --output-dir "$artifact_root/replay"
    ;;
  test)
    "$r16p14_python" -m pytest "$experiment_root/tests" -q
    ;;
  help|*)
    printf '%s\n' \
      'usage: commands.sh {freeze|clue|task-screen|perturbations|actor-cache [args]|actor-smoke|actor-train SEED [DEVICE]|actor-eval SEED [DEVICE]|actor-aggregate|actor-mechanism-audit|atlas-preflight|checkpoint-report|replay|test}'
    ;;
esac
