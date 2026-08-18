from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path(
    os.environ.get("R16_P14_STAGE2D_ARTIFACT_ROOT", PROJECT_ROOT / "artifacts/stage2d")
).resolve()
MIRROR_EXPERIMENT_OUTPUTS = os.environ.get("R16_P14_STAGE2D_MIRROR_EXPERIMENT_OUTPUTS", "1") == "1"

IMMUTABLE_PARENT = "74538ae3d9ff76f1c5c2d981da3a3c133829d73b"
IMMUTABLE_TREE = "93f7f9ae63ff7d2760dc14f65132ee8cdd39e007"
PLAN_SHA256 = "0816abba294faa6e8b12a73660bb819995bb9ee882875496ec795ec87dae8b90"

ACTOR_SEEDS = (7, 17, 29)
TASKS = (
    "put_the_cream_cheese_in_the_bowl",
    "put_the_bowl_on_the_stove",
)
TARGET_SHIFT_TASK = TASKS[0]
PATH_OBSTACLE_TASK = TASKS[1]

INFRASTRUCTURE_IDS = tuple(range(0, 10))
CALIBRATION_IDS = tuple(range(10, 40))
EVALUATION_IDS = tuple(range(40, 80))
RESERVE_IDS = tuple(range(80, 100))
ALL_INIT_IDS = tuple(range(100))
RESET_SEED_BASE = 2_016_214

DETECTION_PREFIX = 2
H_VALID = 16
PREFIX_INDICES = tuple(range(DETECTION_PREFIX, H_VALID + 1))
TAIL_HORIZON = 4
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 216_214
SAFETY_NONINFERIORITY_EPSILON = 0.03

TARGET_SHIFT_GRID = (0.04, 0.06, 0.08)
PATH_FUTURE_INDICES = (DETECTION_PREFIX + 4, DETECTION_PREFIX + 7, DETECTION_PREFIX + 10)
PATH_CLEARANCE_DELTAS = (-0.010, 0.000, 0.010)

CALIBRATION_ARMS = (
    "IMMEDIATE_FRESH",
    "CACHED_MATCHED",
    "FRESH_MATCHED",
    "HOLD_MATCHED",
    "CACHED_NOQUERY",
    "FULL_OLD_CHUNK",
)
CONFIRMATORY_METHODS = (
    "IMMEDIATE_FRESH",
    "FIXED_DELAY_2",
    "FIXED_DELAY_4",
    "FIXED_DELAY_8",
    "EVENT_ALIGNED_CACHED",
    "FRESH_MATCHED_AT_RULE_K",
    "HOLD_MATCHED_AT_RULE_K",
    "CACHED_NOQUERY_AT_RULE_K",
    "FULL_OLD_CHUNK",
)

# The attached plan says to stop after a failed scientific gate. The user's
# explicit request in the chat says every planned experiment must run. We keep
# the gate result immutable, but any downstream execution is diagnostic-only.
FORCED_DIAGNOSTIC_CONTINUATION = True
POSITIVE_LABELS_ALLOWED_AFTER_FAILED_GATE = False
# The outer execution instruction explicitly requires the downstream matrix
# even after a failed scientific gate.  This is an evidence boundary, not a
# gate relaxation: every Stage-2D downstream artifact is diagnostic-only.
DIAGNOSTIC_ONLY_GLOBAL = True
FORMAL_POSITIVE_EVIDENCE_ALLOWED = False
MAXIMUM_GPUS = 2
MAXIMUM_CONCURRENT_GPU_WORKERS = 2
