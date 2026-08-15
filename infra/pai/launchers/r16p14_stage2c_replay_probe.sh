#!/usr/bin/env bash
set -Eeuo pipefail

umask 077

SOURCE_ROOT=/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/r16-p14-stage2c-20260816
EXPERIMENT_ROOT=$SOURCE_ROOT/experiments/r16_p14_stage2c
PYTHON=/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/libero_sft/bin/python
PYTHON_OVERLAY=/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/r22p10-libero-pai-overlay/site-packages
CACHE_ROOT=/mnt/cpfs/zbl-cpfs-new/USERS/leon/cache/r16_p14_stage2c/replay-probe-v1
ARTIFACT_DIR=${PAI_CANARY_RUN_DIR:?PAI_CANARY_RUN_DIR is required}
PROBE_ROOT=$ARTIFACT_DIR/replay_probe
REGISTRY_RUN_ID=${PAI_CANARY_RUN_ID:?PAI_CANARY_RUN_ID is required}
NONCE=${PAI_CANARY_NONCE:?PAI_CANARY_NONCE is required}
EVENTS=/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r16_p14_stage2c/pai/r16p14-stage2c-formal-20260816-v6/stage2c_artifacts/actor_events/events.jsonl
EVENT_ID=put_the_cream_cheese_in_the_bowl__seed07__init30

EXPECTED_SOURCE_COMMIT=ba5ddf68e15fec4db34e38547e8a57721e7fe5ba
EXPECTED_SOURCE_TREE=4f9761c860b05d4d7642ea0275bfa0f658926fc2
EXPECTED_EVENTS_SHA=3fd1d6ede6dddbbd381c9aa749339b46aeef701c795b1047e11b7011b286a495
EXPECTED_PYTHON_SHA=89b2f5166fb529c259aedd43e5f718c60e35d58e630cb40ae6accb48fc4f961a
EXPECTED_OVERLAY_MANIFEST_SHA=64dfffdaf464d1a37be19b038cca919a252dba573eb2d0f8aa442b91a4099459
EXPECTED_OVERLAY_FILE_COUNT=1688
EXPECTED_SEED7_SHA=821177a82cc470e108082fd3c0f6913983236a2fdf142de2fe51fc37c44240ca

on_error() {
  local exit_code=$?
  printf 'R16P14_STAGE2C_REPLAY_PROBE_FAILED line=%s exit_code=%s command=%q\n' \
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
test "$(sha256sum "$EVENTS" | awk '{print $1}')" = "$EXPECTED_EVENTS_SHA"
overlay_manifest_sha=$(find "$PYTHON_OVERLAY" -type f -printf '%P\0' | sort -z | while IFS= read -r -d '' relative; do printf '%s\0' "$relative"; cat "$PYTHON_OVERLAY/$relative"; done | sha256sum | awk '{print $1}')
test "$overlay_manifest_sha" = "$EXPECTED_OVERLAY_MANIFEST_SHA"
test "$(find "$PYTHON_OVERLAY" -type f | wc -l)" = "$EXPECTED_OVERLAY_FILE_COUNT"
test "$(sha256sum "$SOURCE_ROOT/artifacts/stage2a/actor/checkpoints/seed_7.pt" | awk '{print $1}')" = "$EXPECTED_SEED7_SHA"
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -c '^NVIDIA A800')" = 2

export PYTHONPATH="$PYTHON_OVERLAY:$EXPERIMENT_ROOT:$SOURCE_ROOT/experiments/r16_p14_stage2b:$SOURCE_ROOT/experiments/r16_p14_stage2a:$SOURCE_ROOT/experiments/r16_p14_libero_stage1:$SOURCE_ROOT"
export LIBERO_CONFIG_PATH="$SOURCE_ROOT/experiments/r16_p14_libero_stage1/libero_config"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export TORCH_HOME="$CACHE_ROOT/torch"
export HF_HOME="$CACHE_ROOT/huggingface"
export PYTHONPYCACHEPREFIX="$CACHE_ROOT/pycache"
export TMPDIR="$CACHE_ROOT/tmp"

mkdir -p "$PROBE_ROOT" "$XDG_CACHE_HOME" "$TORCH_HOME" "$HF_HOME" "$PYTHONPYCACHEPREFIX" "$TMPDIR"
for slot in 0 1 2; do
  CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
    "$PYTHON" -m r16_p14_stage2c.replay_probe \
      --events "$EVENTS" \
      --event-id "$EVENT_ID" \
      --device cuda \
      --rounds 1 \
      --output "$PROBE_ROOT/slot_${slot}.json" \
      >"$PROBE_ROOT/slot_${slot}.log" 2>&1
  sync -f "$PROBE_ROOT/slot_${slot}.json"
done

"$PYTHON" - <<'PY'
import json
import os
import pathlib
import time

root = pathlib.Path(os.environ["PAI_CANARY_RUN_DIR"]) / "replay_probe"
slots = [json.loads((root / f"slot_{index}.json").read_text()) for index in range(3)]
summary = {
    "schema_version": 1,
    "status": "complete",
    "registry_run_id": os.environ["PAI_CANARY_RUN_ID"],
    "independent_processes": 3,
    "all_passed": all(item["all_passed"] for item in slots),
    "event_id": slots[0]["event_id"],
    "slots": slots,
    "completed_at_unix": time.time(),
}
for name in ("summary.json", "PROBE_COMPLETE.json"):
    path = root / name
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
PY

sync -f "$PROBE_ROOT/PROBE_COMPLETE.json"
test "$(stat -c '%u:%g' "$PROBE_ROOT/PROBE_COMPLETE.json")" = 2254:2254
test "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" = "$EXPECTED_SOURCE_COMMIT"
test "$(git -C "$SOURCE_ROOT" rev-parse 'HEAD^{tree}')" = "$EXPECTED_SOURCE_TREE"
test -z "$(git -C "$SOURCE_ROOT" status --porcelain)"
printf 'R16P14_STAGE2C_REPLAY_PROBE_COMPLETE registry_run_id=%s result=%s\n' "$REGISTRY_RUN_ID" "$PROBE_ROOT/summary.json"
