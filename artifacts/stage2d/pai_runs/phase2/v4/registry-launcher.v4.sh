#!/usr/bin/env bash
set -Eeuo pipefail

export STAGE2D_SOURCE_ROOT=/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/R16-P14-stage2d-fresh-process-event-aligned-prefix-reuse
export STAGE2D_SOURCE_COMMIT=c6ddbdac9044466ace601ef354443177d5168456
export STAGE2D_SOURCE_TREE=fa501143513de15cbec1459baa413b385a943192
export STAGE2D_PHASE=phase2
LAUNCHER="$STAGE2D_SOURCE_ROOT/experiments/r16_p14_stage2d/pai/launcher.sh"
test -x "$LAUNCHER"
exec "$LAUNCHER"
