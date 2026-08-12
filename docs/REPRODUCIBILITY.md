# Reproducibility and artifact map

## Release verification without LIBERO

The integrity validator uses only the Python standard library. It verifies the formal result hashes, row counts and factorial coverage, all nine checkpoint hashes, training metadata, parity records, gate decision, and PAI completion metadata.

```bash
python scripts/verify_r16p14_release.py
sha256sum -c artifacts/SHA256SUMS
```

To persist its result:

```bash
python scripts/verify_r16p14_release.py \
  --output artifacts/test-results/release_verification.json
```

The first command validates the result schema, experimental coverage and key
provenance contracts; the second checks all 84 committed artifact files. These
commands verify the published evidence but do not rerun MuJoCo rollouts.

## Formal source and environment

| Item | Frozen value |
| --- | --- |
| Experiment source commit | `a1b61194a8382f5b1a247b9cd9b140645ff2aeb8` |
| Experiment source tree | `53001c43fbbb165c0a1f2c71f9cbd4c81b9d0ced` |
| LIBERO base | `https://github.com/huggingface/lerobot-libero.git` |
| Python | 3.11 |
| PyTorch | 2.5.1+cu124 |
| CUDA runtime | 12.4 |
| robosuite | 1.4.0 |
| MuJoCo | 3.6.0 |
| Formal hardware | 2× NVIDIA A800 |

The exact PAI contract and command payload are in [`infra/pai`](../infra/pai). They intentionally retain the frozen CPFS mount paths, UID/GID checks, source/tree assertions, data hashes, environment checks, and step/evaluation commands. Credential values are not present.

## Dataset contract

The demonstration datasets are not copied into Git because the three HDF5 files total about 2.05 GB. Place the LIBERO-GOAL files at the paths configured by the experiment, or adapt a private runtime copy of `libero_config/config.yaml` and the dataset root while retaining the hashes below.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `open_the_middle_drawer_of_the_cabinet_demo.hdf5` | 702,223,367 | `20252c7cf98cd7437061f7f200ae7b6cb6219fabbd53b4536dfaa8abda6ab737` |
| `put_the_bowl_on_the_plate_demo.hdf5` | 468,246,288 | `e69528b0cf10dfc59b20698e12ec2affc03f3887309034d3eb74cac3ec929406` |
| `put_the_wine_bottle_on_the_rack_demo.hdf5` | 878,958,730 | `f9092aa70734fc4083e97fc58c3ba25f87c614d18326182ddc7a455f0ab4da2e` |

The exact frozen local LIBERO config has SHA-256 `528990e0cdd466a063def065fddb835fe2f37609cfef305d1910a1bf91a353ce`. It records the content-addressed asset mount used by the formal job.

## Test suite

The release was tested in the existing LIBERO simulation environment with headless EGL:

```bash
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="$PWD:$PWD/experiments/r16_p14_libero_stage1"
export LIBERO_CONFIG_PATH="$PWD/experiments/r16_p14_libero_stage1/libero_config"

python -m pytest experiments/r16_p14_libero_stage1/tests -q \
  --junitxml=artifacts/test-results/pytest.xml
```

The suite covers chunk boundary construction, atomic complete-checkpoint discovery, model output shape, aggregation/gate logic, and deterministic simulator snapshot/suffix replay. The committed JUnit XML records the publication-tree run.

## Formal workload

The exact production launcher is [`infra/pai/launchers/r16p14_libero_stage1.sh`](../infra/pai/launchers/r16p14_libero_stage1.sh). At a high level it performs:

1. immutable source, dataset, environment, mount, GPU, identity, and payload checks;
2. nine 8,000-step state-BC training runs with full-state resume markers;
3. 360 clean baseline rollouts and instrumentation-parity checks;
4. 90 equal-budget oracle branch audits;
5. deterministic aggregation into the preregistered gate decision.

The launcher is cluster-specific and should not be submitted unchanged from an unrelated account or mount layout. See [`infra/pai/README.md`](../infra/pai/README.md) for the provenance and recovery audit.

## Artifact map

| Path | Contents |
| --- | --- |
| `artifacts/formal_pilot/logs/` | stdout/stderr logs for training, baseline, oracle, and aggregation |
| `artifacts/formal_pilot/training/` | per-step JSONL losses, first-real-work marker, training summaries |
| `artifacts/formal_pilot/shards/` | 360 rollout rows, 90 oracle rows, schedules, parity and task summaries |
| `artifacts/formal_pilot/report/` | machine-readable summary, gate decision, generated report |
| `artifacts/checkpoints/` | nine final full-state `checkpoint.pt` files and SHA-256 completion markers |
| `artifacts/test-results/` | tests rerun from this GitHub publication tree |
| `experiments/r16_p14_libero_stage1/source_manifest.json` | frozen source, dataset, job, and key artifact hashes |

Generated caches, temporary files, intermediate step-2000/4000/6000 checkpoints, HDF5 demonstrations, private controller state, and credentials are excluded. Final checkpoints and all formal evaluation evidence are included.
