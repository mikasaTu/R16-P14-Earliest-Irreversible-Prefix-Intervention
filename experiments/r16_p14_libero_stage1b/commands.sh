#!/usr/bin/env bash
set -Eeuo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
experiment_root="$repo_root/experiments/r16_p14_libero_stage1b"
stage1_root="$repo_root/experiments/r16_p14_libero_stage1"
r16p14_python=${R16P14_PYTHON:-python}
export PYTHONPATH="$repo_root:$stage1_root:$experiment_root${PYTHONPATH:+:$PYTHONPATH}"

case "${1:-}" in
  phase-a)
    "$r16p14_python" -m r16_p14_stage1b.offline_reanalysis \
      --stage1-artifacts "$repo_root/artifacts/formal_pilot" \
      --output-dir "$repo_root/artifacts/stage1b/offline_reanalysis"
    ;;
  phase-b)
    "$r16p14_python" -m r16_p14_stage1b.replay_reconstruction \
      --repo-root "$repo_root" \
      --output-dir "$repo_root/artifacts/stage1b/replay_gate" \
      --device "${R16P14_REPLAY_DEVICE:-cuda}"
    ;;
  phase-c-calibrate)
    CUDA_VISIBLE_DEVICES='' "$r16p14_python" -m r16_p14_stage1b.expert_chunk_calibration \
      --repo-root "$repo_root" \
      --output-dir "$repo_root/artifacts/stage1b/expert_chunk_calibration"
    ;;
  phase-c-summarize)
    "$r16p14_python" -m r16_p14_stage1b.expert_gate \
      --source-dir "$repo_root/artifacts/stage1b/expert_chunk_calibration"
    ;;
  test)
    "$r16p14_python" -m pytest "$experiment_root/tests" -q
    ;;
  verify)
    "$r16p14_python" -m r16_p14_stage1b.release \
      --repo-root "$repo_root" --write-report --write-manifest
    ;;
  *)
    printf 'usage: %s {phase-a|phase-b|phase-c-calibrate|phase-c-summarize|test|verify}\n' "$0" >&2
    exit 2
    ;;
esac
