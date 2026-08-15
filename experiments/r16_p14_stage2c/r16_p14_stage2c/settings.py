from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path(
    os.environ.get("R16_P14_STAGE2C_ARTIFACT_ROOT", PROJECT_ROOT / "artifacts/stage2c")
).resolve()

ACTOR_SEEDS = (7, 17, 29)
CALIBRATION_IDS = tuple(range(30, 40))
EVALUATION_IDS = tuple(range(40, 50))
DETECTION_PREFIX = 2
H_VALID = 16
PREFIX_INDICES = tuple(range(DETECTION_PREFIX, H_VALID + 1))
TAIL_HORIZON = 16
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 160_214

TARGET_SHIFT_TASK = "put_the_cream_cheese_in_the_bowl"
PATH_TASK_CANDIDATES = (
    "put_the_bowl_on_the_stove",
    "push_the_plate_to_the_front_of_the_stove",
    "open_the_top_drawer_and_put_the_bowl_inside",
)
ALL_CANDIDATE_TASKS = (TARGET_SHIFT_TASK,) + PATH_TASK_CANDIDATES
TARGET_SHIFT_GRID = (0.04, 0.06, 0.08)
PATH_INDEX_GRID = (DETECTION_PREFIX + 4, DETECTION_PREFIX + 8, DETECTION_PREFIX + 12)
PATH_CLEARANCE_GRID = (0.00, 0.02, 0.04)

GOAL_SITE_HINTS = {
    "put_the_bowl_on_the_stove": "flat_stove_1_cook_region",
    "push_the_plate_to_the_front_of_the_stove": "main_table_stove_front_region",
    "open_the_top_drawer_and_put_the_bowl_inside": "wooden_cabinet_1_top_region",
}

RECOVERY_OPERATORS = (
    "fresh_h16",
    "fresh_h4",
    "hold_one_step_then_fresh_h16",
    "rollback_one_step_then_fresh_h16",
)

