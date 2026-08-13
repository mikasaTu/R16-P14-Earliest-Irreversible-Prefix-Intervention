# Stage-1b first checkpoint — Phase A + Bowl replay repair

Checkpoint date: 2026-08-13 UTC

This checkpoint separates metric correction from replay correctness. It is not evidence for a learned detector, deployable intervention policy, or final algorithm.

## 1. Exact frozen HEAD/tree

- Current `main` at freeze: commit `ee56aa096e308214c38132d0e6d2a9e576c29792`, tree `50e56b84c5e867447b22f81e31454646a12a9eb8`.
- Original Stage-1 frozen source: commit `a1b61194a8382f5b1a247b9cd9b140645ff2aeb8`, tree `53001c43fbbb165c0a1f2c71f9cbd4c81b9d0ced`.
- Work branch: `agent/stage1b-contract-repair` in independent worktree `/workspace/leon/R16-P14-stage1b`.
- Stage-1 source and original artifacts were not modified.

## 2. Corrected 90-record reanalysis

All 90 immutable Stage-1 candidate records were reread offline. No simulator or GPU was started for Phase A. The corrected primary label counts only an explicit target cause; task non-completion and timeout remain separate fields.

## 3. Old versus corrected metrics

| Task | Old mixed unsafe | Explicit cause violations | Failure-only inclusions | Old median window | Corrected median window | Old retention | Corrected post-detection retention | Complete M0/M1/M2 pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Drawer | 27 | 6 | 21 | 0 | 0 | 0% | 0% | 0 |
| Bowl | 17 | 5 | 12 | 4 | 0 | 50% | 0% | 2 |
| Wine | 28 | 28 | 0 | 0 | 0 | 0% | 0% | 4 |

The old metric implicitly treated failure to finish inside the budget as a safety violation. The corrected metric does not. Retention is now `(k_last_safe - d) / (chunk_length - d)`, rather than `k_last_safe / chunk_length`.

## 4. Does the Bowl positive signal survive?

No. Only 5/30 Bowl candidates have an explicit nominal cause violation, and just 2 form complete M0/M1/M2 comparisons. Their M0/M1/M2 cause-violation counts are 0/1/1, so the intervention does not beat immediate full replanning on safety. The median corrected window and median post-detection retention are both zero.

M1 uses 41.5 median newly executed non-nominal actions versus 60 for M0 on this two-pair subset, but the subset is too small and M1 has worse safety. This is recorded only as a weak rework observation, not a positive result.

## 5. Snapshot restore versus prefix reconstruction

The old initializer restores a branch snapshot directly. The repaired initializer creates a fresh independent environment, restores the demonstration phase anchor, replays the exact nominal prefix, injects the recorded perturbation at `d`, and then continues to each branch point. Five independent reconstruction environments plus a separate snapshot-control environment are used; mutable branch state is never shared across operators or repetitions.

| Initializer | Insertion-point pass | All branch-point pass | Contact/outcome agreement | Max final-state absolute error |
| --- | ---: | ---: | ---: | ---: |
| Published Stage-1 snapshot restore | 26/30 (86.7%) | not recorded | not recorded | not recorded |
| Snapshot-restore control, 5 repeats | 26/30 (86.7%) | 147/180 (81.7%) | 149/180 (82.8%) | 3.7161693483379317 |
| Fresh-env prefix reconstruction, 5 repeats | 30/30 (100%) | 180/180 (100%) | 180/180 (100%) | 0 |

Only 57/180 reconstructed branch hashes equal the old Stage-1 snapshot hash. This is expected evidence that the old initializer omitted trajectory/controller history; it is not a failure of the new reconstruction contract.

## 6. Replay gate result

**PASS.** The preregistered gate requires at least 99% branch-point agreement, 100% contact/outcome agreement, and maximum final-state error no greater than `1e-9`. Reconstruction achieves 100%, 100%, and 0 respectively across 30 candidates, six branch points per candidate, and five repetitions per branch point. Exact Stage-1 action-chunk hashes match for 30/30 candidates.

One local A800 was used only for frozen MLP forward passes because the Stage-1 action byte hashes were created with PyTorch 2.5.1+cu124 CUDA; CPU inference changes floating-point bytes. Simulation and replay execution are CPU-side. No training occurred.

## 7. Exact files changed at this checkpoint

New code and contracts:

- `experiments/r16_p14_libero_stage1b/README.md`
- `experiments/r16_p14_libero_stage1b/preregistration.yaml`
- `experiments/r16_p14_libero_stage1b/metric_contract.md`
- `experiments/r16_p14_libero_stage1b/source_manifest.json`
- `experiments/r16_p14_libero_stage1b/decision.json`
- `experiments/r16_p14_libero_stage1b/commands.sh`
- `experiments/r16_p14_libero_stage1b/r16_p14_stage1b/__init__.py`
- `experiments/r16_p14_libero_stage1b/r16_p14_stage1b/offline_reanalysis.py`
- `experiments/r16_p14_libero_stage1b/r16_p14_stage1b/replay_reconstruction.py`
- `experiments/r16_p14_libero_stage1b/tests/test_offline_reanalysis.py`
- `experiments/r16_p14_libero_stage1b/tests/test_replay_reconstruction.py`
- `experiments/r16_p14_libero_stage1b/reports/FIRST_CHECKPOINT.md`

New immutable outputs:

- `artifacts/stage1b/offline_reanalysis/{reanalysis.jsonl,summary.json,paired_metrics.csv,old_to_new_metrics.csv,report.md}`
- `artifacts/stage1b/replay_gate/{branch_reconstructions.jsonl,summary.json,report.md}`

## 8. Blocker

There is no replay or infrastructure blocker. The scientific signal is currently negative after metric correction, so progress beyond Phase B is conditional on a new, disjoint expert-action audit. Phases D and E remain prohibited until their upstream gates pass.

## 9. Next exact command

After the Phase C runner and frozen calibration grid are code-reviewed and tested, the next scientific command is:

```bash
cd /workspace/leon/R16-P14-stage1b
R16P14_PYTHON=/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/libero_sft/bin/python \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
./experiments/r16_p14_libero_stage1b/commands.sh phase-c-calibrate
```

This command calibrates revised environment-only perturbations on demo IDs 0–9 only. It does not train a model and may not inspect evaluation or held-out demonstrations for parameter selection.
