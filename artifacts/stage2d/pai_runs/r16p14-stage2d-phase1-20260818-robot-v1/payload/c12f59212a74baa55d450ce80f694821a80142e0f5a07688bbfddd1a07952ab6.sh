#!/usr/bin/env bash
set -Eeuo pipefail

export STAGE2D_SOURCE_ROOT=/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/r16-p14-stage2d-20260818-phase1
export STAGE2D_SOURCE_COMMIT=65c0092b5b4aa27b79c0d54a99aad9bcd30eaa64
export STAGE2D_SOURCE_TREE=5f3e47a982cdeaec52d326f942a94421d0635323
export STAGE2D_PHASE=phase1
LAUNCHER="$STAGE2D_SOURCE_ROOT/experiments/r16_p14_stage2d/pai/launcher.sh"
test -x "$LAUNCHER"
exec "$LAUNCHER"
