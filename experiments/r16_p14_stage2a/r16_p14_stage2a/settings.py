from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/stage2a"
DATASET_ROOT = Path("/mnt/cpfs/zbl-cpfs-new/dataset/leon/libero/libero_goal")
DEFAULT_LIBERO_CONFIG = (
    PROJECT_ROOT / "experiments/r16_p14_libero_stage1/libero_config"
)
DEFAULT_CACHE_ROOT = Path(
    "/mnt/cpfs/zbl-cpfs-new/USERS/leon/cache/r16_p14_stage2a"
)
DEFAULT_CKPT_ROOT = Path(
    "/mnt/cpfs/zbl-cpfs-new/CKPT/leon/torch/r16_p14_stage2a"
)
DEFAULT_LOG_ROOT = Path(
    "/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r16_p14_stage2a"
)

FEATURE_KEYS = ("robot0_proprio-state", "object-state")
FEATURE_DIM = 95
ACTION_DIM = 7
CHUNK_LENGTH = 16
INSERTION_PREFIX = 2
OBS_HISTORY = 4
ACTION_HISTORY = 3
TRAIN_SEEDS = (7, 17, 29)
CALIBRATION_DEMOS = tuple(range(0, 10))
EVALUATION_DEMOS = tuple(range(10, 40))
HELD_OUT_DEMOS = tuple(range(40, 50))


@dataclass(frozen=True)
class TaskSpec:
    task_id: int
    name: str
    hdf5_name: str
    family: str
    horizon: int
    manipulated_joint: str | None = None
    target_joint: str | None = None
    obstacle_joint: str | None = None
    mechanism_joint: str | None = None
    handle_geom: str | None = None
    lift_delta: float = 0.03
    placement_xy_tolerance: float = 0.08

    @property
    def hdf5_path(self) -> Path:
        return DATASET_ROOT / self.hdf5_name


TASK_SPECS: dict[str, TaskSpec] = {
    "put_the_cream_cheese_in_the_bowl": TaskSpec(
        task_id=6,
        name="put_the_cream_cheese_in_the_bowl",
        hdf5_name="put_the_cream_cheese_in_the_bowl_demo.hdf5",
        family="target_shift",
        horizon=360,
        manipulated_joint="cream_cheese_1_joint0",
        target_joint="akita_black_bowl_1_joint0",
        lift_delta=0.025,
        placement_xy_tolerance=0.060,
    ),
    "put_the_bowl_on_the_plate": TaskSpec(
        task_id=8,
        name="put_the_bowl_on_the_plate",
        hdf5_name="put_the_bowl_on_the_plate_demo.hdf5",
        family="target_shift",
        horizon=320,
        manipulated_joint="akita_black_bowl_1_joint0",
        target_joint="plate_1_joint0",
        lift_delta=0.030,
        placement_xy_tolerance=0.080,
    ),
    "open_the_middle_drawer_of_the_cabinet": TaskSpec(
        task_id=0,
        name="open_the_middle_drawer_of_the_cabinet",
        hdf5_name="open_the_middle_drawer_of_the_cabinet_demo.hdf5",
        family="drawer_obstacle",
        horizon=360,
        obstacle_joint="wine_bottle_1_joint0",
        mechanism_joint="wooden_cabinet_1_middle_level",
        handle_geom="wooden_cabinet_1_g29",
    ),
    "open_the_top_drawer_and_put_the_bowl_inside": TaskSpec(
        task_id=3,
        name="open_the_top_drawer_and_put_the_bowl_inside",
        hdf5_name="open_the_top_drawer_and_put_the_bowl_inside_demo.hdf5",
        family="drawer_obstacle",
        horizon=620,
        manipulated_joint="akita_black_bowl_1_joint0",
        obstacle_joint="wine_bottle_1_joint0",
        mechanism_joint="wooden_cabinet_1_top_level",
        handle_geom="wooden_cabinet_1_g18",
        lift_delta=0.030,
    ),
    "push_the_plate_to_the_front_of_the_stove": TaskSpec(
        task_id=5,
        name="push_the_plate_to_the_front_of_the_stove",
        hdf5_name="push_the_plate_to_the_front_of_the_stove_demo.hdf5",
        family="swept_path_blocker",
        horizon=420,
        manipulated_joint="plate_1_joint0",
        obstacle_joint="cream_cheese_1_joint0",
    ),
    "put_the_bowl_on_the_stove": TaskSpec(
        task_id=1,
        name="put_the_bowl_on_the_stove",
        hdf5_name="put_the_bowl_on_the_stove_demo.hdf5",
        family="target_region_blocker",
        horizon=360,
        manipulated_joint="akita_black_bowl_1_joint0",
        obstacle_joint="plate_1_joint0",
        lift_delta=0.030,
        placement_xy_tolerance=0.090,
    ),
}

TASK_NAMES = tuple(TASK_SPECS)
TASK_TO_INDEX = {name: index for index, name in enumerate(TASK_NAMES)}


def perturbation_grid(task_name: str) -> list[dict[str, float | int]]:
    family = TASK_SPECS[task_name].family
    if family == "target_shift":
        return [
            {"release_lead_actions": lead, "shift_magnitude_m": magnitude}
            for lead in (8, 10, 12, 14)
            for magnitude in (0.04, 0.06, 0.08, 0.10)
        ]
    if family == "drawer_obstacle":
        return [
            {"insertion_offset": offset, "obstacle_clearance_m": clearance}
            for offset in (-2, 0, 2)
            for clearance in (0.035, 0.05, 0.065, 0.08)
        ]
    if family == "target_region_blocker":
        return [
            {"release_lead_actions": lead, "blocker_lateral_offset_m": offset}
            for lead in (8, 10, 12, 14)
            for offset in (0.00, 0.025, 0.050, 0.075)
        ]
    if family == "swept_path_blocker":
        return [
            {
                "target_path_progress_fraction": progress,
                "blocker_lateral_offset_m": offset,
            }
            for progress in (0.70, 0.80, 0.90, 1.00)
            for offset in (0.00, 0.025, 0.050, 0.075)
        ]
    raise KeyError(family)


def config_id(task_name: str, parameters: dict[str, float | int]) -> str:
    family = TASK_SPECS[task_name].family
    if family == "target_shift":
        return (
            f"lead{int(parameters['release_lead_actions']):02d}_"
            f"shift{round(float(parameters['shift_magnitude_m']) * 1000):03d}mm"
        )
    if family == "drawer_obstacle":
        return (
            f"offset{int(parameters['insertion_offset']):+03d}_"
            f"clear{round(float(parameters['obstacle_clearance_m']) * 1000):03d}mm"
        )
    if family == "target_region_blocker":
        return (
            f"lead{int(parameters['release_lead_actions']):02d}_"
            f"lateral{round(float(parameters['blocker_lateral_offset_m']) * 1000):03d}mm"
        )
    return (
        f"progress{round(float(parameters['target_path_progress_fraction']) * 100):03d}_"
        f"lateral{round(float(parameters['blocker_lateral_offset_m']) * 1000):03d}mm"
    )
