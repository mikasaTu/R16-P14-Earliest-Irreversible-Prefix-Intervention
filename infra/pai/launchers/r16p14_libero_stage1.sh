#!/usr/bin/env bash
set -Eeuo pipefail

umask 077

SOURCE_ROOT=/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/r16-p14-libero-stage1-20260812
EXPERIMENT_ROOT=$SOURCE_ROOT/experiments/r16_p14_libero_stage1
PYTHON=/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/libero_sft/bin/python
PYTHON_OVERLAY=/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/r22p10-libero-pai-overlay/site-packages
DATA_ROOT=/mnt/cpfs/zbl-cpfs-new/dataset/leon/libero/libero_goal
ASSET_ROOT=/mnt/cpfs/zbl-cpfs-new/dataset/leon/libero/assets/90001343cb134b7e26e18fde0fa2416f3ed6e6a3
CHECKPOINT_BASE=/mnt/cpfs/zbl-cpfs-new/CKPT/leon/torch/r16_p14_libero_stage1
APPLICATION_RUN_ID=r16p14-libero-stage1-pilot-v1
CHECKPOINT_LINEAGE=$CHECKPOINT_BASE/$APPLICATION_RUN_ID
CACHE_ROOT=/mnt/cpfs/zbl-cpfs-new/USERS/leon/cache/r16_p14_libero_stage1/pilot-v1
ARTIFACT_DIR=${PAI_CANARY_RUN_DIR:?PAI_CANARY_RUN_DIR is required}
REGISTRY_RUN_ID=${PAI_CANARY_RUN_ID:?PAI_CANARY_RUN_ID is required}
NONCE=${PAI_CANARY_NONCE:?PAI_CANARY_NONCE is required}

EXPECTED_SOURCE_COMMIT=a1b61194a8382f5b1a247b9cd9b140645ff2aeb8
EXPECTED_SOURCE_TREE=53001c43fbbb165c0a1f2c71f9cbd4c81b9d0ced
EXPECTED_LIBERO_CONFIG_SHA=528990e0cdd466a063def065fddb835fe2f37609cfef305d1910a1bf91a353ce
EXPECTED_DRAWER_SHA=20252c7cf98cd7437061f7f200ae7b6cb6219fabbd53b4536dfaa8abda6ab737
EXPECTED_BOWL_SHA=e69528b0cf10dfc59b20698e12ec2affc03f3887309034d3eb74cac3ec929406
EXPECTED_WINE_SHA=f9092aa70734fc4083e97fc58c3ba25f87c614d18326182ddc7a455f0ab4da2e
EXPECTED_PYTHON_SHA=89b2f5166fb529c259aedd43e5f718c60e35d58e630cb40ae6accb48fc4f961a
EXPECTED_OVERLAY_MANIFEST_SHA=64dfffdaf464d1a37be19b038cca919a252dba573eb2d0f8aa442b91a4099459
EXPECTED_OVERLAY_FILE_COUNT=1688

on_error() {
  local exit_code=$?
  printf 'R16P14_FORMAL_COMMAND_FAILED line=%s exit_code=%s command=%q\n' \
    "${BASH_LINENO[0]:-unknown}" "$exit_code" "$BASH_COMMAND" >&2
  return "$exit_code"
}
trap on_error ERR

for required in git sha256sum nvidia-smi stat realpath awk grep find sort wc cat sync; do
  command -v "$required" >/dev/null
done
test "$(id -u):$(id -g)" = 2254:2254
test "${PAI_CANARY_EXPECTED_GPUS:-}" = 2
test "${WANDB_ENTITY:-}" = chen_jian-cj-workspace
test -n "${WANDB_API_KEY:-}"
[[ "$REGISTRY_RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{2,63}$ ]]
[[ "$NONCE" =~ ^[a-f0-9]{32}$ ]]
case "$ARTIFACT_DIR" in
  /mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r16_p14_libero_stage1/pai/*) ;;
  *) printf 'artifact directory escaped the R16-P14 root\n' >&2; exit 71 ;;
esac
test "$(realpath -e "$ARTIFACT_DIR")" = "$ARTIFACT_DIR"
test "$(stat -c '%u:%g' "$ARTIFACT_DIR")" = 2254:2254
test "$(stat -c '%u:%g' "$CHECKPOINT_LINEAGE")" = 2254:2254
test "$(stat -c '%u:%g' "$CACHE_ROOT")" = 2254:2254
cd "$ARTIFACT_DIR"
test "$(sha256sum "$PAI_MOUNT_SENTINEL" | awk '{print $1}')" = "$PAI_MOUNT_SENTINEL_SHA256"

test "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" = "$EXPECTED_SOURCE_COMMIT"
test "$(git -C "$SOURCE_ROOT" rev-parse 'HEAD^{tree}')" = "$EXPECTED_SOURCE_TREE"
test -z "$(git -C "$SOURCE_ROOT" status --porcelain)"
test "$(sha256sum "$EXPERIMENT_ROOT/libero_config/config.yaml" | awk '{print $1}')" = "$EXPECTED_LIBERO_CONFIG_SHA"
test "$(sha256sum "$DATA_ROOT/open_the_middle_drawer_of_the_cabinet_demo.hdf5" | awk '{print $1}')" = "$EXPECTED_DRAWER_SHA"
test "$(sha256sum "$DATA_ROOT/put_the_bowl_on_the_plate_demo.hdf5" | awk '{print $1}')" = "$EXPECTED_BOWL_SHA"
test "$(sha256sum "$DATA_ROOT/put_the_wine_bottle_on_the_rack_demo.hdf5" | awk '{print $1}')" = "$EXPECTED_WINE_SHA"
test -f "$ASSET_ROOT/scenes/libero_tabletop_base_style.xml"
test -x "$PYTHON"
test "$(sha256sum "$(realpath -e "$PYTHON")" | awk '{print $1}')" = "$EXPECTED_PYTHON_SHA"
overlay_manifest_sha=$(find "$PYTHON_OVERLAY" -type f -printf '%P\0' | sort -z | while IFS= read -r -d '' relative; do printf '%s\0' "$relative"; cat "$PYTHON_OVERLAY/$relative"; done | sha256sum | awk '{print $1}')
test "$overlay_manifest_sha" = "$EXPECTED_OVERLAY_MANIFEST_SHA"
test "$(find "$PYTHON_OVERLAY" -type f | wc -l)" = "$EXPECTED_OVERLAY_FILE_COUNT"
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -c '^NVIDIA A800')" = 2

export PYTHONPATH="$PYTHON_OVERLAY:$SOURCE_ROOT:$EXPERIMENT_ROOT"
export R16P14_SOURCE_ROOT="$SOURCE_ROOT"
export LIBERO_CONFIG_PATH="$EXPERIMENT_ROOT/libero_config"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export XDG_CACHE_HOME="$ARTIFACT_DIR/cache/xdg"
export TORCH_HOME="$ARTIFACT_DIR/cache/torch"
export PYTHONPYCACHEPREFIX="$ARTIFACT_DIR/cache/pycache"
export WANDB_DIR="$ARTIFACT_DIR/cache/wandb"
export R16P14_WANDB_PROJECT=r16-p14-libero-stage1
export TMPDIR="$ARTIFACT_DIR/tmp"
mkdir -p \
  "$XDG_CACHE_HOME" "$TORCH_HOME" "$PYTHONPYCACHEPREFIX" "$WANDB_DIR" \
  "$TMPDIR" "$ARTIFACT_DIR/logs" "$ARTIFACT_DIR/training" \
  "$ARTIFACT_DIR/shards" "$ARTIFACT_DIR/report"

"$PYTHON" - <<'PY'
import os
import pathlib
import torch
import libero

source = pathlib.Path(os.environ["R16P14_SOURCE_ROOT"])
assert os.getuid() == 2254 and os.getgid() == 2254
assert torch.__version__ == "2.5.1+cu124" and torch.version.cuda == "12.4"
assert torch.cuda.device_count() == 2
assert pathlib.Path(libero.__file__).resolve().is_relative_to(source)
print(
    {
        "python": os.sys.executable,
        "torch": torch.__version__,
        "gpus": torch.cuda.device_count(),
        "libero": libero.__file__,
    },
    flush=True,
)
PY

printf 'PAI_AUTO_FAULT_TOLERANCE=0 APPLICATION_RUN_ID=%s CHECKPOINT_LINEAGE=%s\n' \
  "$APPLICATION_RUN_ID" "$CHECKPOINT_LINEAGE"

run_train_task() {
  local task_name=$1
  local gpu_index=$2
  local log_path="$ARTIFACT_DIR/logs/train-${task_name}.log"
  CUDA_VISIBLE_DEVICES="$gpu_index" MUJOCO_EGL_DEVICE_ID="$gpu_index" \
    "$PYTHON" -m r16_p14_stage1.train \
      --tasks "$task_name" \
      --seeds 7 17 29 \
      --run-id "$APPLICATION_RUN_ID" \
      --cache-root "$CACHE_ROOT" \
      --checkpoint-root "$CHECKPOINT_BASE" \
      --log-root "$ARTIFACT_DIR/training" \
      --libero-config "$EXPERIMENT_ROOT/libero_config" \
      --demo-count 50 \
      --chunk-length 16 \
      --hidden-dim 512 \
      --batch-size 256 \
      --steps 8000 \
      --save-interval 2000 \
      --learning-rate 0.0003 \
      --device cuda >"$log_path" 2>&1
}

run_train_task open_the_middle_drawer_of_the_cabinet 0 &
drawer_train_pid=$!
run_train_task put_the_bowl_on_the_plate 1 &
bowl_train_pid=$!

set +e
wait "$drawer_train_pid"
drawer_train_status=$?
wait "$bowl_train_pid"
bowl_train_status=$?
set -e
test "$drawer_train_status" = 0
test "$bowl_train_status" = 0

run_train_task put_the_wine_bottle_on_the_rack 0

"$PYTHON" - <<PY
import json
import os
import pathlib
import time

from r16_p14_stage1.io_utils import atomic_write_json

checkpoint = pathlib.Path("$CHECKPOINT_LINEAGE")
artifact = pathlib.Path("$ARTIFACT_DIR")
tasks = [
    "open_the_middle_drawer_of_the_cabinet",
    "put_the_bowl_on_the_plate",
    "put_the_wine_bottle_on_the_rack",
]
seeds = [7, 17, 29]
results = []
for task in tasks:
    for seed in seeds:
        path = checkpoint / task / f"seed_{seed}" / ".training_complete.json"
        value = json.loads(path.read_text())
        assert value["step"] == 8000
        results.append(value)
atomic_write_json(
    artifact / "TRAINING_COMPLETE.json",
    {
        "status": "complete",
        "application_run_id": "$APPLICATION_RUN_ID",
        "registry_run_id": "$REGISTRY_RUN_ID",
        "results": results,
        "uid": os.getuid(),
        "gid": os.getgid(),
        "completed_at_unix": time.time(),
    },
)
PY
sync -f "$ARTIFACT_DIR/TRAINING_COMPLETE.json"
test "$(stat -c '%u:%g' "$ARTIFACT_DIR/TRAINING_COMPLETE.json")" = 2254:2254
printf 'R16P14_TRAINING_COMPLETE application_run_id=%s\n' "$APPLICATION_RUN_ID"

run_eval_task() {
  local task_name=$1
  local gpu_index=$2
  local shard_root="$ARTIFACT_DIR/shards/$task_name"
  local baseline_log="$ARTIFACT_DIR/logs/baseline-${task_name}.log"
  local oracle_log="$ARTIFACT_DIR/logs/oracle-${task_name}.log"
  mkdir -p "$shard_root"
  CUDA_VISIBLE_DEVICES="$gpu_index" MUJOCO_EGL_DEVICE_ID="$gpu_index" \
    "$PYTHON" -m r16_p14_stage1.eval_baseline \
      --run-id "$APPLICATION_RUN_ID" \
      --tasks "$task_name" \
      --seeds 7 17 29 \
      --horizons 1 4 8 16 \
      --episodes 10 \
      --checkpoint-root "$CHECKPOINT_BASE" \
      --log-root "$shard_root" \
      --libero-config "$EXPERIMENT_ROOT/libero_config" \
      --device cuda \
      --parity \
      --parity-steps 32 >"$baseline_log" 2>&1
  CUDA_VISIBLE_DEVICES="$gpu_index" MUJOCO_EGL_DEVICE_ID="$gpu_index" \
    "$PYTHON" -m r16_p14_stage1.oracle_audit \
      --run-id "$APPLICATION_RUN_ID" \
      --task "$task_name" \
      --model-seed 7 \
      --demo-count 10 \
      --max-candidates 30 \
      --insertion-prefixes 2 6 10 \
      --prefix-stride 2 \
      --branch-budget 96 \
      --execution-horizon 8 \
      --checkpoint-root "$CHECKPOINT_BASE" \
      --log-root "$shard_root" \
      --libero-config "$EXPERIMENT_ROOT/libero_config" \
      --device cuda >"$oracle_log" 2>&1
}

run_eval_task open_the_middle_drawer_of_the_cabinet 0 &
drawer_eval_pid=$!
run_eval_task put_the_bowl_on_the_plate 1 &
bowl_eval_pid=$!

set +e
wait "$drawer_eval_pid"
drawer_eval_status=$?
wait "$bowl_eval_pid"
bowl_eval_status=$?
set -e
test "$drawer_eval_status" = 0
test "$bowl_eval_status" = 0

run_eval_task put_the_wine_bottle_on_the_rack 0

CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m r16_p14_stage1.aggregate \
  --run-id "$APPLICATION_RUN_ID" \
  --shard-root "$ARTIFACT_DIR/shards" \
  --checkpoint-root "$CHECKPOINT_BASE" \
  --output-dir "$ARTIFACT_DIR/report" \
  --baseline-episodes 10 \
  --oracle-candidates 30 >"$ARTIFACT_DIR/logs/aggregate.log" 2>&1
cp "$ARTIFACT_DIR/report/EVALUATION_COMPLETE.json" "$ARTIFACT_DIR/EVALUATION_COMPLETE.json"
sync -f "$ARTIFACT_DIR/EVALUATION_COMPLETE.json"
test "$(stat -c '%u:%g' "$ARTIFACT_DIR/EVALUATION_COMPLETE.json")" = 2254:2254
printf 'R16P14_FORMAL_COMPLETE registry_run_id=%s report=%s\n' \
  "$REGISTRY_RUN_ID" "$ARTIFACT_DIR/report/report.md"
