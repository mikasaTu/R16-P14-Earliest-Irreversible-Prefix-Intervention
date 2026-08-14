from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/stage2b"
STAGE2A_EXPERIMENT_ROOT = PROJECT_ROOT / "experiments/r16_p14_stage2a"
STAGE2A_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/stage2a"
LIBERO_CONFIG = PROJECT_ROOT / "experiments/r16_p14_libero_stage1/libero_config"

PYTHON = Path("/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/libero_sim/bin/python")
PRIMARY_TASKS = (
    "put_the_cream_cheese_in_the_bowl",
    "put_the_bowl_on_the_stove",
)
ACTOR_SEEDS = (7, 17, 29)
EXECUTION_HORIZONS = (1, 2, 4, 8, 16)
CHUNK_LENGTH = 16
DETECTION_PREFIX = 2
OBS_HISTORY = 4
ACTION_HISTORY = 3
ACTION_DIM = 7

ACTOR_QUALIFICATION_IDS = tuple(range(0, 10))
PERTURBATION_CALIBRATION_IDS = tuple(range(10, 20))
ATLAS_EVALUATION_IDS = tuple(range(20, 30))
RESERVED_IDS = tuple(range(30, 50))
ALL_INIT_IDS = tuple(range(50))

CHECKPOINTS = {
    seed: STAGE2A_ARTIFACT_ROOT / f"actor/checkpoints/seed_{seed}.pt"
    for seed in ACTOR_SEEDS
}

STAGE1_COMMIT = "ee56aa096e308214c38132d0e6d2a9e576c29792"
STAGE1_TREE = "50e56b84c5e867447b22f81e31454646a12a9eb8"
STAGE1B_COMMIT = "e29e3ead42fd1799b412a4968e6a67aac3784874"
STAGE1B_TREE = "4016b05942e6dfb291f1bc3a2644e177a208b608"
STAGE2A_EVIDENCE_COMMIT = "df20e31cefc4db22ba23f2b61284469e781d5558"
STAGE2A_EVIDENCE_TREE = "bfe056abce8550cfc3ece69cfd3198f690728dfa"
STAGE2A_REPORT_COMMIT = "dd86afbb445169ecef6eb25fb0a73d09763585b5"
STAGE2A_REPORT_TREE = "e69f4e468ee64e2ab7ceaed330c0dad32265e89f"

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 260814
POST_DETECTION_BUDGET_CAP = 200

# The user explicitly overrode the attachment's early-stop rule. Gates retain
# their scientific meaning, but every later phase is executed and labelled.
CONTINUE_AFTER_GATE_FAILURE = True
FALLBACK_ANALYSIS_HORIZON = 8

CREAM_GRID_M = (0.03, 0.04, 0.05, 0.06)
STOVE_GRID_M = (0.04, 0.05, 0.06, 0.075)
STAGE2A_FROZEN_MAGNITUDES = {
    "put_the_cream_cheese_in_the_bowl": (0.04, 0.06),
    "put_the_bowl_on_the_stove": (0.05, 0.075),
}

TASK_LABELS = {
    "put_the_cream_cheese_in_the_bowl": "cream-cheese -> bowl",
    "put_the_bowl_on_the_stove": "bowl -> stove",
}
