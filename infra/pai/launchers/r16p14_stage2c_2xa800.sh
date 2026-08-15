#!/usr/bin/env bash
set -Eeuo pipefail

umask 077

SOURCE_ROOT=/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/r16-p14-stage2c-20260816
EXPERIMENT_ROOT=$SOURCE_ROOT/experiments/r16_p14_stage2c
PYTHON=/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/libero_sft/bin/python
PYTHON_OVERLAY=/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/r22p10-libero-pai-overlay/site-packages
APPLICATION_RUN_ID=r16p14-stage2c-formal-20260816-v1
CACHE_ROOT=/mnt/cpfs/zbl-cpfs-new/USERS/leon/cache/r16_p14_stage2c/formal-v1
ARTIFACT_DIR=${PAI_CANARY_RUN_DIR:?PAI_CANARY_RUN_DIR is required}
ARTIFACT_ROOT=$ARTIFACT_DIR/stage2c_artifacts
REGISTRY_RUN_ID=${PAI_CANARY_RUN_ID:?PAI_CANARY_RUN_ID is required}
NONCE=${PAI_CANARY_NONCE:?PAI_CANARY_NONCE is required}

EXPECTED_SOURCE_COMMIT=47e9648da463487bd3c35d5ea55d56ec15be7848
EXPECTED_SOURCE_TREE=910edceb0ab24a00f266af6d35c29f3ee778a87a
EXPECTED_PYTHON_SHA=89b2f5166fb529c259aedd43e5f718c60e35d58e630cb40ae6accb48fc4f961a
EXPECTED_OVERLAY_MANIFEST_SHA=64dfffdaf464d1a37be19b038cca919a252dba573eb2d0f8aa442b91a4099459
EXPECTED_OVERLAY_FILE_COUNT=1688
EXPECTED_LIBERO_CONFIG_SHA=528990e0cdd466a063def065fddb835fe2f37609cfef305d1910a1bf91a353ce
EXPECTED_SEED7_SHA=821177a82cc470e108082fd3c0f6913983236a2fdf142de2fe51fc37c44240ca
EXPECTED_SEED17_SHA=83ee61e31ffdae6f2ef57203a2c0085df41e284039cd81c6ecf1210694521604
EXPECTED_SEED29_SHA=0cf34a3e535525345306a2b322aae3b1bd6ebd6cd71dc653e2a91393e2b79d1a

on_error() {
  local exit_code=$?
  printf 'R16P14_STAGE2C_COMMAND_FAILED line=%s exit_code=%s command=%q\n' \
    "${BASH_LINENO[0]:-unknown}" "$exit_code" "$BASH_COMMAND" >&2
  return "$exit_code"
}
trap on_error ERR

for required in git sha256sum nvidia-smi stat realpath awk grep find sort sync cat wc; do
  command -v "$required" >/dev/null
done
test "$(id -u):$(id -g)" = 2254:2254
test "${PAI_CANARY_EXPECTED_GPUS:-}" = 2
[[ "$REGISTRY_RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{2,63}$ ]]
[[ "$NONCE" =~ ^[a-f0-9]{32}$ ]]
case "$ARTIFACT_DIR" in
  /mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r16_p14_stage2c/pai/*) ;;
  *) printf 'artifact directory escaped the R16-P14 Stage2C root\n' >&2; exit 71 ;;
esac
test "$(realpath -e "$ARTIFACT_DIR")" = "$ARTIFACT_DIR"
test "$(stat -c '%u:%g' "$ARTIFACT_DIR")" = 2254:2254
test "$(stat -c '%u:%g' "$CACHE_ROOT")" = 2254:2254
test "$(sha256sum "$PAI_MOUNT_SENTINEL" | awk '{print $1}')" = "$PAI_MOUNT_SENTINEL_SHA256"
cd "$ARTIFACT_DIR"

test "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" = "$EXPECTED_SOURCE_COMMIT"
test "$(git -C "$SOURCE_ROOT" rev-parse 'HEAD^{tree}')" = "$EXPECTED_SOURCE_TREE"
test -z "$(git -C "$SOURCE_ROOT" status --porcelain)"
test -x "$PYTHON"
test "$(sha256sum "$(realpath -e "$PYTHON")" | awk '{print $1}')" = "$EXPECTED_PYTHON_SHA"
overlay_manifest_sha=$(find "$PYTHON_OVERLAY" -type f -printf '%P\0' | sort -z | while IFS= read -r -d '' relative; do printf '%s\0' "$relative"; cat "$PYTHON_OVERLAY/$relative"; done | sha256sum | awk '{print $1}')
test "$overlay_manifest_sha" = "$EXPECTED_OVERLAY_MANIFEST_SHA"
test "$(find "$PYTHON_OVERLAY" -type f | wc -l)" = "$EXPECTED_OVERLAY_FILE_COUNT"
test "$(sha256sum "$SOURCE_ROOT/experiments/r16_p14_libero_stage1/libero_config/config.yaml" | awk '{print $1}')" = "$EXPECTED_LIBERO_CONFIG_SHA"
test "$(sha256sum "$SOURCE_ROOT/artifacts/stage2a/actor/checkpoints/seed_7.pt" | awk '{print $1}')" = "$EXPECTED_SEED7_SHA"
test "$(sha256sum "$SOURCE_ROOT/artifacts/stage2a/actor/checkpoints/seed_17.pt" | awk '{print $1}')" = "$EXPECTED_SEED17_SHA"
test "$(sha256sum "$SOURCE_ROOT/artifacts/stage2a/actor/checkpoints/seed_29.pt" | awk '{print $1}')" = "$EXPECTED_SEED29_SHA"
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -c '^NVIDIA A800')" = 2

export PYTHONPATH="$PYTHON_OVERLAY:$EXPERIMENT_ROOT:$SOURCE_ROOT/experiments/r16_p14_stage2b:$SOURCE_ROOT/experiments/r16_p14_stage2a:$SOURCE_ROOT/experiments/r16_p14_libero_stage1:$SOURCE_ROOT"
export R16_P14_STAGE2C_ARTIFACT_ROOT="$ARTIFACT_ROOT"
export R16_P14_STAGE2C_MIRROR_EXPERIMENT_OUTPUTS=0
export LIBERO_CONFIG_PATH="$SOURCE_ROOT/experiments/r16_p14_libero_stage1/libero_config"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export TORCH_HOME="$CACHE_ROOT/torch"
export HF_HOME="$CACHE_ROOT/huggingface"
export PYTHONPYCACHEPREFIX="$CACHE_ROOT/pycache"
export TMPDIR="$CACHE_ROOT/tmp"

mkdir -p "$ARTIFACT_ROOT" "$ARTIFACT_DIR/logs" "$XDG_CACHE_HOME" "$TORCH_HOME" "$HF_HOME" "$PYTHONPYCACHEPREFIX" "$TMPDIR"
if [[ ! -d "$ARTIFACT_ROOT/contract_repair" ]]; then
  cp -a "$SOURCE_ROOT/artifacts/stage2c/contract_repair" "$ARTIFACT_ROOT/contract_repair"
fi
if [[ ! -d "$ARTIFACT_ROOT/source_freeze" ]]; then
  cp -a "$EXPERIMENT_ROOT/source_freeze" "$ARTIFACT_ROOT/source_freeze"
fi
if [[ ! -f "$ARTIFACT_ROOT/preregistration.yaml" ]]; then
  cp "$EXPERIMENT_ROOT/preregistration.yaml" "$ARTIFACT_ROOT/preregistration.yaml"
fi
if [[ ! -f "$ARTIFACT_ROOT/metric_contract.md" ]]; then
  cp "$EXPERIMENT_ROOT/metric_contract.md" "$ARTIFACT_ROOT/metric_contract.md"
fi

"$PYTHON" - <<'PY'
import os
import pathlib
import torch

from r16_p14_stage2b.runtime import ActorBundle
from r16_p14_stage2c.settings import ARTIFACT_ROOT

assert os.getuid() == 2254 and os.getgid() == 2254
assert torch.cuda.is_available() and torch.cuda.device_count() == 2
assert torch.__version__ == "2.5.1+cu124" and torch.version.cuda == "12.4"
assert pathlib.Path(ARTIFACT_ROOT).is_dir()
for seed in (7, 17, 29):
    assert ActorBundle.load(seed, "cpu").seed == seed
print({"torch": torch.__version__, "cuda": torch.version.cuda, "gpus": torch.cuda.device_count(), "artifact_root": str(ARTIFACT_ROOT)}, flush=True)
PY

run_gpu() {
  local gpu_index=$1
  local log_path=$2
  shift 2
  CUDA_VISIBLE_DEVICES="$gpu_index" MUJOCO_EGL_DEVICE_ID=0 \
    "$PYTHON" "$@" >>"$log_path" 2>&1
}

run_event_worker() {
  local worker_index=$1
  local gpu_index=$2
  local log_path="$ARTIFACT_DIR/logs/events-worker${worker_index}.log"
  local index=0
  local task seed split
  local tasks=(put_the_cream_cheese_in_the_bowl put_the_bowl_on_the_stove push_the_plate_to_the_front_of_the_stove open_the_top_drawer_and_put_the_bowl_inside)
  for split in calibration evaluation; do
    for task in "${tasks[@]}"; do
      for seed in 7 17 29; do
        if (( index % 2 == worker_index )); then
          run_gpu "$gpu_index" "$log_path" -m r16_p14_stage2c.events --seed "$seed" --task "$task" --split "$split" --device cuda
        fi
        index=$((index + 1))
      done
    done
  done
}

run_event_worker 0 0 &
event_pid0=$!
run_event_worker 1 1 &
event_pid1=$!
wait "$event_pid0"
wait "$event_pid1"
sync -f "$ARTIFACT_ROOT/first_completed_event.json"
if [[ ! -f "$ARTIFACT_ROOT/actor_events/summary.json" ]]; then
  "$PYTHON" -m r16_p14_stage2c.events --consolidate >"$ARTIFACT_DIR/logs/events-consolidate.log" 2>&1
fi

run_qualification_worker() {
  local worker_index=$1
  local gpu_index=$2
  local log_path="$ARTIFACT_DIR/logs/qualification-worker${worker_index}.log"
  local index=0
  local task seed
  local tasks=(put_the_cream_cheese_in_the_bowl put_the_bowl_on_the_stove push_the_plate_to_the_front_of_the_stove open_the_top_drawer_and_put_the_bowl_inside)
  for task in "${tasks[@]}"; do
    for seed in 7 17 29; do
      if (( index % 2 == worker_index )); then
        run_gpu "$gpu_index" "$log_path" -m r16_p14_stage2c.qualification --seed "$seed" --task "$task" --device cuda
      fi
      index=$((index + 1))
    done
  done
}

run_qualification_worker 0 0 &
qualification_pid0=$!
run_qualification_worker 1 1 &
qualification_pid1=$!
wait "$qualification_pid0"
wait "$qualification_pid1"
if [[ ! -f "$ARTIFACT_ROOT/task_qualification/summary.json" ]]; then
  "$PYTHON" -m r16_p14_stage2c.qualification --consolidate >"$ARTIFACT_DIR/logs/qualification-consolidate.log" 2>&1
fi
if [[ ! -f "$ARTIFACT_ROOT/actor_events/formal_event_pool_summary.json" ]]; then
  "$PYTHON" -m r16_p14_stage2c.evaluation --prepare-pool >"$ARTIFACT_DIR/logs/prepare-pool.log" 2>&1
fi

if [[ ! -f "$ARTIFACT_ROOT/integration_smoke/summary.json" ]]; then
  run_gpu 0 "$ARTIFACT_DIR/logs/integration-smoke.log" -m r16_p14_stage2c.evaluation --smoke --device cuda --worker-count 1 --worker-index 0
  "$PYTHON" -m r16_p14_stage2c.evaluation --smoke --consolidate >>"$ARTIFACT_DIR/logs/integration-smoke.log" 2>&1
fi

if [[ ! -f "$ARTIFACT_ROOT/formal_matrix/summary.json" ]]; then
  run_gpu 0 "$ARTIFACT_DIR/logs/evaluation-worker0.log" -m r16_p14_stage2c.evaluation --device cuda --worker-count 2 --worker-index 0 &
  evaluation_pid0=$!
  run_gpu 1 "$ARTIFACT_DIR/logs/evaluation-worker1.log" -m r16_p14_stage2c.evaluation --device cuda --worker-count 2 --worker-index 1 &
  evaluation_pid1=$!
  wait "$evaluation_pid0"
  wait "$evaluation_pid1"
  "$PYTHON" -m r16_p14_stage2c.evaluation --consolidate >"$ARTIFACT_DIR/logs/evaluation-consolidate.log" 2>&1
fi

if [[ ! -f "$ARTIFACT_ROOT/decision.json" ]]; then
  "$PYTHON" -m r16_p14_stage2c.aggregate --bootstrap-replicates 10000 >"$ARTIFACT_DIR/logs/aggregate.log" 2>&1
fi
if [[ ! -f "$ARTIFACT_ROOT/REPORT.md" ]]; then
  "$PYTHON" -m r16_p14_stage2c.report >"$ARTIFACT_DIR/logs/report.log" 2>&1
fi

if [[ ! -f "$ARTIFACT_ROOT/STAGE2C_COMPLETE.json" ]]; then
  "$PYTHON" - <<'PY'
import json
import os
import pathlib
import time

root = pathlib.Path(os.environ["R16_P14_STAGE2C_ARTIFACT_ROOT"])
payload = {
    "schema_version": 1,
    "status": "completed_all_planned_attempts",
    "application_run_id": "r16p14-stage2c-formal-20260816-v1",
    "registry_run_id": os.environ["PAI_CANARY_RUN_ID"],
    "uid": os.getuid(),
    "gid": os.getgid(),
    "formal_matrix": json.loads((root / "formal_matrix/summary.json").read_text()),
    "decision": json.loads((root / "decision.json").read_text()),
    "completed_at_unix": time.time(),
}
path = root / "STAGE2C_COMPLETE.json"
descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(descriptor, "w") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
fi
sync -f "$ARTIFACT_ROOT/STAGE2C_COMPLETE.json"
test "$(stat -c '%u:%g' "$ARTIFACT_ROOT/STAGE2C_COMPLETE.json")" = 2254:2254
test "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" = "$EXPECTED_SOURCE_COMMIT"
test "$(git -C "$SOURCE_ROOT" rev-parse 'HEAD^{tree}')" = "$EXPECTED_SOURCE_TREE"
test -z "$(git -C "$SOURCE_ROOT" status --porcelain)"
printf 'R16P14_STAGE2C_COMPLETE registry_run_id=%s report=%s\n' "$REGISTRY_RUN_ID" "$ARTIFACT_ROOT/REPORT.md"
