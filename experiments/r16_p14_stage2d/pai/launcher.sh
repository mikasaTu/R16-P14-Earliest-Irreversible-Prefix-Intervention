#!/usr/bin/env bash
set -Eeuo pipefail

umask 077

SOURCE_ROOT=${STAGE2D_SOURCE_ROOT:?STAGE2D_SOURCE_ROOT is required}
EXPECTED_SOURCE_COMMIT=${STAGE2D_SOURCE_COMMIT:?STAGE2D_SOURCE_COMMIT is required}
EXPECTED_SOURCE_TREE=${STAGE2D_SOURCE_TREE:?STAGE2D_SOURCE_TREE is required}
PHASE=${STAGE2D_PHASE:?STAGE2D_PHASE is required}
PYTHON=/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/libero_sim/bin/python
ARTIFACT_DIR=${PAI_CANARY_RUN_DIR:?PAI_CANARY_RUN_DIR is required}
ARTIFACT_ROOT=$ARTIFACT_DIR/stage2d_artifacts
CACHE_ROOT=/mnt/cpfs/zbl-cpfs-new/USERS/leon/cache/r16_p14_stage2d
REGISTRY_RUN_ID=${PAI_CANARY_RUN_ID:?PAI_CANARY_RUN_ID is required}
NONCE=${PAI_CANARY_NONCE:?PAI_CANARY_NONCE is required}
RESUME_FROM_RUN=${STAGE2D_RESUME_FROM_RUN:-}
RESUME_SOURCE_COMMIT=${STAGE2D_RESUME_SOURCE_COMMIT:-$EXPECTED_SOURCE_COMMIT}
RESUME_SOURCE_TREE=${STAGE2D_RESUME_SOURCE_TREE:-$EXPECTED_SOURCE_TREE}
RESUME_ROOT=${STAGE2D_RESUME_ROOT:-/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r16_p14_stage2d/pai}

on_error() {
  local exit_code=$?
  printf 'R16P14_STAGE2D_COMMAND_FAILED phase=%s line=%s exit_code=%s command=%q\n' \
    "$PHASE" "${BASH_LINENO[0]:-unknown}" "$exit_code" "$BASH_COMMAND" >&2
  return "$exit_code"
}
trap on_error ERR

for required in git sha256sum nvidia-smi stat realpath awk grep find sort sync; do
  command -v "$required" >/dev/null
done
test "$(id -u):$(id -g)" = 2254:2254
test "${PAI_CANARY_EXPECTED_GPUS:-}" = 2
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -c '^NVIDIA A800')" = 2
[[ "$REGISTRY_RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{2,63}$ ]]
[[ "$NONCE" =~ ^[a-f0-9]{32}$ ]]
case "$PHASE" in phase1|atlas|phase2|phase3) ;; *) printf 'unknown Stage2D phase: %s\n' "$PHASE" >&2; exit 64 ;; esac
case "$ARTIFACT_DIR" in
  /mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r16_p14_stage2d/pai/*) ;;
  *) printf 'artifact directory escaped Stage2D root\n' >&2; exit 71 ;;
esac
test "$(realpath -e "$ARTIFACT_DIR")" = "$ARTIFACT_DIR"
test "$(stat -c '%u:%g' "$ARTIFACT_DIR")" = 2254:2254
test "$(stat -c '%u:%g' "$CACHE_ROOT")" = 2254:2254
test "$(sha256sum "$PAI_MOUNT_SENTINEL" | awk '{print $1}')" = "$PAI_MOUNT_SENTINEL_SHA256"
test "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" = "$EXPECTED_SOURCE_COMMIT"
test "$(git -C "$SOURCE_ROOT" rev-parse 'HEAD^{tree}')" = "$EXPECTED_SOURCE_TREE"
test -z "$(git -C "$SOURCE_ROOT" status --porcelain)"
test -x "$PYTHON"

# PAI starts the outer bootstrap in /root.  The bootstrap drops privileges to
# UID/GID 2254 before this launcher runs, so multiprocessing.spawn cannot
# reconstruct a child from that unreadable inherited cwd.  Bind every phase to
# the already hash-verified source checkout before any Python process starts.
cd -- "$SOURCE_ROOT"
test "$(pwd -P)" = "$SOURCE_ROOT"

export PYTHONPATH="$SOURCE_ROOT/experiments/r16_p14_stage2d:$SOURCE_ROOT/experiments/r16_p14_stage2c:$SOURCE_ROOT/experiments/r16_p14_stage2b:$SOURCE_ROOT/experiments/r16_p14_stage2a:$SOURCE_ROOT/experiments/r16_p14_libero_stage1:$SOURCE_ROOT"
export R16_P14_STAGE2D_ARTIFACT_ROOT="$ARTIFACT_ROOT"
export R16_P14_STAGE2D_MIRROR_EXPERIMENT_OUTPUTS=0
export LIBERO_CONFIG_PATH="$SOURCE_ROOT/experiments/r16_p14_libero_stage1/libero_config"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export TORCH_HOME="$CACHE_ROOT/torch"
export HF_HOME="$CACHE_ROOT/huggingface"
export PYTHONPYCACHEPREFIX="$CACHE_ROOT/pycache"
export TMPDIR="$CACHE_ROOT/tmp"

mkdir -p "$ARTIFACT_ROOT" "$ARTIFACT_DIR/logs" "$XDG_CACHE_HOME" "$TORCH_HOME" "$HF_HOME" "$PYTHONPYCACHEPREFIX" "$TMPDIR"
if [[ ! -f "$ARTIFACT_ROOT/source_freeze/manifest.json" ]]; then
  cp -a "$SOURCE_ROOT/artifacts/stage2d/." "$ARTIFACT_ROOT/"
fi
test -f "$ARTIFACT_ROOT/source_freeze/manifest.json"
test -f "$ARTIFACT_ROOT/init_pool/init_states.npz"
test -f "$ARTIFACT_ROOT/branch_isolation/summary.json"

run_worker() {
  local gpu_index=$1
  local log_path=$2
  shift 2
  CUDA_VISIBLE_DEVICES="$gpu_index" MUJOCO_EGL_DEVICE_ID="$gpu_index" \
    "$PYTHON" "$@" >>"$log_path" 2>&1
}

write_marker() {
  local marker_name=$1
  "$PYTHON" - "$marker_name" "$PHASE" <<'PY'
import json, os, pathlib, sys, time
root = pathlib.Path(os.environ["R16_P14_STAGE2D_ARTIFACT_ROOT"])
path = root.parent / "pai_state" / sys.argv[1]
path.parent.mkdir(parents=True, exist_ok=True)
payload = {
    "schema_version": 1,
    "status": "complete",
    "phase": sys.argv[2],
    "registry_run_id": os.environ["PAI_CANARY_RUN_ID"],
    "uid": os.getuid(),
    "gid": os.getgid(),
    "unix_time": time.time(),
}
if not path.exists():
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
PY
}

write_terminal_receipt() {
  local relative_path=$1
  "$PYTHON" - "$relative_path" "$PHASE" <<'PY'
import json, os, pathlib, sys, time
root = pathlib.Path(os.environ["R16_P14_STAGE2D_ARTIFACT_ROOT"])
path = root / sys.argv[1]
path.parent.mkdir(parents=True, exist_ok=True)
payload = {
    "schema_version": 1,
    "phase": sys.argv[2],
    "status": "SUCCEEDED",
    "complete_matrix": True,
    "workers_returned_zero": True,
    "registry_run_id": os.environ["PAI_CANARY_RUN_ID"],
    "source": "terminal worker barrier before consolidation",
    "unix_time": time.time(),
}
if not path.exists():
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
PY
}

import_resume_shards() {
  [[ -n "$RESUME_FROM_RUN" ]] || return 0
  [[ "$PHASE" = phase2 ]] || { printf 'resume import is phase2-only\n' >&2; exit 73; }
  [[ "$RESUME_FROM_RUN" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{2,63}$ ]]
  if [[ "${STAGE2D_RESUME_TEST_MODE:-0}" != 1 ]]; then
    case "$RESUME_ROOT" in
      /mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r16_p14_stage2d/pai) ;;
      *) printf 'resume root escaped Stage2D evidence root: %s\n' "$RESUME_ROOT" >&2; exit 72 ;;
    esac
  fi
  local resume_dir="$RESUME_ROOT/$RESUME_FROM_RUN"
  test -f "$resume_dir/pai_state/RECOVERY_MANIFEST.json"
  "$PYTHON" - "$resume_dir" "$ARTIFACT_ROOT" "$RESUME_SOURCE_COMMIT" "$RESUME_SOURCE_TREE" <<'PY'
import hashlib
import json
import os
import pathlib
import shutil
import sys

resume_dir = pathlib.Path(sys.argv[1])
artifact_dir = pathlib.Path(sys.argv[2])
expected_commit, expected_tree = sys.argv[3:5]
manifest_path = resume_dir / "pai_state/RECOVERY_MANIFEST.json"
manifest = json.loads(manifest_path.read_text())
if not manifest.get("immutable_completed_evidence"):
    raise RuntimeError("resume manifest is not immutable evidence")
if manifest.get("source_commit") != expected_commit or manifest.get("source_tree") != expected_tree:
    raise RuntimeError("resume source freeze differs from current source")
source_root = resume_dir / "stage2d_artifacts"
destination_root = artifact_dir / "confirmatory_evaluation"
destination_root.mkdir(parents=True, exist_ok=True)
imported = []
for item in manifest.get("items", []):
    relative = pathlib.PurePosixPath(item["relative_path"])
    if relative.parts[:2] != ("stage2d_artifacts", "confirmatory_evaluation"):
        raise RuntimeError(f"resume path escaped confirmatory root: {relative}")
    source = resume_dir / relative
    destination = artifact_dir / relative.relative_to("stage2d_artifacts")
    if not source.is_file():
        raise RuntimeError(f"resume shard is missing: {source}")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != item["sha256"] or source.stat().st_size != int(item["size"]):
        raise RuntimeError(f"resume shard checksum mismatch: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"refusing to overwrite divergent completed shard: {destination}")
    else:
        temporary = destination.with_name(
            f".{destination.name}.{os.getpid()}.tmp"
        )
        descriptor = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        try:
            with source.open("rb") as source_handle, os.fdopen(descriptor, "wb") as destination_handle:
                shutil.copyfileobj(source_handle, destination_handle)
                destination_handle.flush()
                os.fsync(destination_handle.fileno())
            try:
                os.link(temporary, destination)
            except FileExistsError:
                if hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
                    raise RuntimeError(
                        f"refusing to overwrite divergent concurrent shard: {destination}"
                    )
            else:
                directory_descriptor = os.open(destination.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_descriptor)
                finally:
                    os.close(directory_descriptor)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    imported.append({"relative_path": str(destination.relative_to(artifact_dir)), "sha256": digest})
receipt = artifact_dir.parent / "pai_state/RESUME_IMPORT.json"
receipt.parent.mkdir(parents=True, exist_ok=True)
payload = {
    "schema_version": 1,
    "status": "IMPORTED_IMMUTABLE_COMPLETED_SHARDS",
    "source_run_id": manifest["run_id"],
    "source_job_id": manifest["job_id"],
    "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    "source_replay_shard_count": len(imported),
    "expected_replay_shard_count": manifest["expected_replay_shard_count"],
    "source_status": manifest["status"],
    "uid": os.getuid(),
    "gid": os.getgid(),
    "completed_shards_not_overwritten": True,
}
if not receipt.exists():
    descriptor = os.open(receipt, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
PY
}

run_event_worker() {
  local worker_index=$1
  local gpu_index=$2
  local log_path="$ARTIFACT_DIR/logs/events-worker${worker_index}.log"
  local index=0 task seed split
  local tasks=(put_the_cream_cheese_in_the_bowl put_the_bowl_on_the_stove)
  for split in calibration evaluation; do
    for task in "${tasks[@]}"; do
      for seed in 7 17 29; do
        if (( index % 2 == worker_index )); then
          run_worker "$gpu_index" "$log_path" -m r16_p14_stage2d.events \
            --seed "$seed" --task "$task" --split "$split" --device cpu
        fi
        index=$((index + 1))
      done
    done
  done
}

run_dual_module() {
  local module=$1
  shift
  run_worker 0 "$ARTIFACT_DIR/logs/${module##*.}-worker0.log" -m "$module" \
    --worker-count 2 --worker-index 0 --device cpu "$@" &
  local pid0=$!
  run_worker 1 "$ARTIFACT_DIR/logs/${module##*.}-worker1.log" -m "$module" \
    --worker-count 2 --worker-index 1 --device cpu "$@" &
  local pid1=$!
  wait "$pid0"
  wait "$pid1"
}

import_resume_shards

if [[ "$PHASE" = phase1 ]]; then
  # One persisted formal event is the first-real-work gate. All later work is
  # resume-safe at immutable JSON shard boundaries; no training checkpoint is applicable.
  run_worker 0 "$ARTIFACT_DIR/logs/first-event.log" -m r16_p14_stage2d.events \
    --task put_the_cream_cheese_in_the_bowl --seed 7 --split calibration --init-id 10 --device cpu
  write_marker FIRST_REAL_WORK.json
  sync -f "$ARTIFACT_DIR/pai_state/FIRST_REAL_WORK.json"

  run_event_worker 0 0 &
  event_pid0=$!
  run_event_worker 1 1 &
  event_pid1=$!
  wait "$event_pid0"
  wait "$event_pid1"
  "$PYTHON" -m r16_p14_stage2d.events --consolidate \
    --consolidate-splits calibration,evaluation >"$ARTIFACT_DIR/logs/events-consolidate.log" 2>&1

  run_dual_module r16_p14_stage2d.qualification
  "$PYTHON" -m r16_p14_stage2d.qualification --consolidate >"$ARTIFACT_DIR/logs/qualification-consolidate.log" 2>&1
  "$PYTHON" -m r16_p14_stage2d.cohort >"$ARTIFACT_DIR/logs/cohort.log" 2>&1
  write_marker PHASE1_COMPLETE.json
elif [[ "$PHASE" = atlas ]]; then
  test -f "$ARTIFACT_ROOT/checkpoint_barrier.json"
  test -f "$ARTIFACT_ROOT/actor_events/formal_event_pool.jsonl"
  run_worker 0 "$ARTIFACT_DIR/logs/first-calibration-branch.log" \
    -m r16_p14_stage2d.calibration --worker-count 1 --worker-index 0 \
    --device cpu --max-new-shards 1
  write_marker FIRST_REAL_WORK.json
  run_dual_module r16_p14_stage2d.calibration
  write_marker ATLAS_WORKERS_COMPLETE.json
  write_terminal_receipt calibration_atlas/terminal_receipt.json
  "$PYTHON" -m r16_p14_stage2d.calibration --consolidate >"$ARTIFACT_DIR/logs/calibration-consolidate.log" 2>&1
  write_marker ATLAS_COMPLETE.json
elif [[ "$PHASE" = phase2 ]]; then
  test -f "$ARTIFACT_ROOT/frozen_rule/manifest.json"
  run_worker 0 "$ARTIFACT_DIR/logs/first-confirmatory-replay.log" \
    -m r16_p14_stage2d.confirmatory --phase replay --worker-count 1 \
    --worker-index 0 --device cpu --max-new-shards 1
  write_marker FIRST_REAL_WORK.json
  run_dual_module r16_p14_stage2d.confirmatory --phase replay
  run_dual_module r16_p14_stage2d.confirmatory --phase methods
  write_marker PHASE2_WORKERS_COMPLETE.json
  write_terminal_receipt confirmatory_evaluation/terminal_receipt.json
  "$PYTHON" -m r16_p14_stage2d.confirmatory --consolidate >"$ARTIFACT_DIR/logs/confirmatory-consolidate.log" 2>&1
  "$PYTHON" -m r16_p14_stage2d.statistics --primary >"$ARTIFACT_DIR/logs/statistics-primary.log" 2>&1
  "$PYTHON" -m r16_p14_stage2d.mechanism_reverse >"$ARTIFACT_DIR/logs/mechanism-primary.log" 2>&1
  write_marker PHASE2_COMPLETE.json
else
  test -f "$ARTIFACT_ROOT/statistics/primary_manifest.json"
  run_worker 0 "$ARTIFACT_DIR/logs/first-oracle-appendix.log" \
    -m r16_p14_stage2d.oracle_appendix --worker-count 1 --worker-index 0 \
    --device cpu --max-new-shards 1
  write_marker FIRST_REAL_WORK.json
  run_dual_module r16_p14_stage2d.oracle_appendix
  write_marker PHASE3_WORKERS_COMPLETE.json
  write_terminal_receipt oracle_appendix/terminal_receipt.json
  "$PYTHON" -m r16_p14_stage2d.oracle_appendix --consolidate >"$ARTIFACT_DIR/logs/oracle-consolidate.log" 2>&1
  "$PYTHON" -m r16_p14_stage2d.mechanism_reverse >"$ARTIFACT_DIR/logs/mechanism-final.log" 2>&1
  "$PYTHON" -m r16_p14_stage2d.report >"$ARTIFACT_DIR/logs/report.log" 2>&1
  "$PYTHON" -m pytest -q "$SOURCE_ROOT/experiments/r16_p14_stage2d/tests" \
    --junitxml="$ARTIFACT_ROOT/test_results/pytest.xml" >"$ARTIFACT_DIR/logs/pytest.log" 2>&1
  "$PYTHON" -m r16_p14_stage2d.checksums >"$ARTIFACT_DIR/logs/checksums.log" 2>&1
  "$PYTHON" -m pytest -q "$SOURCE_ROOT/experiments/r16_p14_stage2d/tests" \
    >"$ARTIFACT_DIR/logs/pytest-final.log" 2>&1
  write_marker PHASE3_COMPLETE.json
fi

completion_marker=${PHASE^^}_COMPLETE.json
sync -f "$ARTIFACT_DIR/pai_state/$completion_marker"
test "$(stat -c '%u:%g' "$ARTIFACT_DIR/pai_state/$completion_marker")" = 2254:2254
test "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" = "$EXPECTED_SOURCE_COMMIT"
test "$(git -C "$SOURCE_ROOT" rev-parse 'HEAD^{tree}')" = "$EXPECTED_SOURCE_TREE"
test -z "$(git -C "$SOURCE_ROOT" status --porcelain)"
printf 'R16P14_STAGE2D_COMPLETE phase=%s registry_run_id=%s artifact_root=%s\n' \
  "$PHASE" "$REGISTRY_RUN_ID" "$ARTIFACT_ROOT"
