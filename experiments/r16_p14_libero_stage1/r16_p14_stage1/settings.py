from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path("/mnt/cpfs/zbl-cpfs-new/dataset/leon/libero/libero_goal")
DEFAULT_CACHE_ROOT = Path(
    "/mnt/cpfs/zbl-cpfs-new/USERS/leon/cache/r16_p14_libero_stage1"
)
DEFAULT_LOG_ROOT = Path(
    "/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r16_p14_libero_stage1"
)
DEFAULT_CKPT_ROOT = Path(
    "/mnt/cpfs/zbl-cpfs-new/CKPT/leon/torch/r16_p14_libero_stage1"
)
DEFAULT_LIBERO_CONFIG = EXPERIMENT_ROOT / "libero_config"

FEATURE_KEYS = ("robot0_proprio-state", "object-state")
CHUNK_LENGTH = 16
EXECUTION_HORIZONS = (1, 4, 8, 16)
TRAIN_SEEDS = (7, 17, 29)


@dataclass(frozen=True)
class TaskSpec:
    task_id: int
    name: str
    hdf5_name: str
    cause: str
    max_episode_steps: int
    perturbation_joint: str
    perturbation_axis: int

    @property
    def hdf5_path(self) -> Path:
        return DATASET_ROOT / self.hdf5_name


TASK_SPECS = {
    "open_the_middle_drawer_of_the_cabinet": TaskSpec(
        task_id=0,
        name="open_the_middle_drawer_of_the_cabinet",
        hdf5_name="open_the_middle_drawer_of_the_cabinet_demo.hdf5",
        cause="mechanism_path_obstacle",
        max_episode_steps=320,
        perturbation_joint="wine_bottle_1_joint0",
        perturbation_axis=1,
    ),
    "put_the_bowl_on_the_plate": TaskSpec(
        task_id=8,
        name="put_the_bowl_on_the_plate",
        hdf5_name="put_the_bowl_on_the_plate_demo.hdf5",
        cause="target_shift_or_premature_release",
        max_episode_steps=320,
        perturbation_joint="plate_1_joint0",
        perturbation_axis=0,
    ),
    "put_the_wine_bottle_on_the_rack": TaskSpec(
        task_id=9,
        name="put_the_wine_bottle_on_the_rack",
        hdf5_name="put_the_wine_bottle_on_the_rack_demo.hdf5",
        cause="grasp_slip_or_contact_misalignment",
        max_episode_steps=520,
        perturbation_joint="wine_bottle_1_joint0",
        perturbation_axis=0,
    ),
    "open_the_top_drawer_and_put_the_bowl_inside": TaskSpec(
        task_id=3,
        name="open_the_top_drawer_and_put_the_bowl_inside",
        hdf5_name="open_the_top_drawer_and_put_the_bowl_inside_demo.hdf5",
        cause="held_out_composed_mechanism_and_release",
        max_episode_steps=620,
        perturbation_joint="akita_black_bowl_1_joint0",
        perturbation_axis=0,
    ),
}

DEVELOPMENT_TASKS = tuple(list(TASK_SPECS)[:3])
HELD_OUT_TASKS = ("open_the_top_drawer_and_put_the_bowl_inside",)
