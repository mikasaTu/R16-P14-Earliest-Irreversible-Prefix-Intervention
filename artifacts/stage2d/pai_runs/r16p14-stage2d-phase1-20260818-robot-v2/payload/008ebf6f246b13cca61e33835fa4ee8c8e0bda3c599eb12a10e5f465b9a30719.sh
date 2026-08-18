#!/usr/bin/env bash
set -Eeuo pipefail

export STAGE2D_SOURCE_ROOT=/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/r16-p14-stage2d-20260818-phase1-recovery-v2
export STAGE2D_SOURCE_COMMIT=5254032b41c76294acf7988aaa7836e74df4e70c
export STAGE2D_SOURCE_TREE=f1de7c52dc90faafde2ff4b1e54d6a1bf5967e0b
export STAGE2D_PHASE=phase1
LAUNCHER="$STAGE2D_SOURCE_ROOT/experiments/r16_p14_stage2d/pai/launcher.sh"
test -x "$LAUNCHER"
exec "$LAUNCHER"
