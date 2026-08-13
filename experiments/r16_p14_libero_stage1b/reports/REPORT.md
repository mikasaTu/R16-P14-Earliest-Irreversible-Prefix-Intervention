# R16-P14 Stage-1b final report

## Decision

**`KILL_CORE_HYPOTHESIS`**

Stage-1b repairs two important experimental contracts but does not establish the proposed mechanism across tasks. The corrected metric removes the published Stage-1 Bowl positive. Fresh-environment prefix reconstruction repairs Bowl replay to 100%. The subsequent policy-free expert calibration yields a valid intervention window on only one bounded replacement task; the preregistered expert gate requires at least two tasks. The failure is already monotonic at calibration, so evaluation-data retuning, policy training, and the revised policy oracle are prohibited.

This result does not say that all prefix intervention is useless. It says the preregistered R16-P14 cross-task core hypothesis did not survive its contract-repair and expert-chunk feasibility gate.

## Frozen evidence boundary

- Parent `main` at freeze: commit `ee56aa096e308214c38132d0e6d2a9e576c29792`, tree `50e56b84c5e867447b22f81e31454646a12a9eb8`.
- Original Stage-1 source: commit `a1b61194a8382f5b1a247b9cd9b140645ff2aeb8`, tree `53001c43fbbb165c0a1f2c71f9cbd4c81b9d0ced`.
- Independent Stage-1b branch/worktree: `agent/stage1b-contract-repair`, `/workspace/leon/R16-P14-stage1b`.
- Stage-1 source, raw records, clean rollouts, and checkpoints were read-only inputs and were not modified.
- Stage-1b used LIBERO-GOAL, state observations, JSONL/CSV/Markdown, ordinary Git, and SHA256.
- No learned risk head, π0.5, RGB model, world model, new policy training, or final-algorithm performance experiment was run.

Raw-input and runtime hashes are in `source_manifest.json`; all Stage-1b output hashes are in `artifacts/stage1b/SHA256SUMS`.

## Phase A — metric correction

Phase A reread all 90 immutable Stage-1 oracle candidate records offline. It did not start a simulator or GPU.

The corrected contract separates `cause_violation`, `task_failure`, `timeout`, and `safe_success`. A task that fails to finish within its budget is no longer automatically a safety violation. The primary retention is now:

`(k_last_safe - d) / (chunk_length - d)`

instead of `k_last_safe / chunk_length`.

| Task | Old mixed unsafe | Explicit cause violations | Failure-only inclusions | Old median window | Corrected median window | Old retention | Corrected post-detection retention | Complete M0/M1/M2 pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Drawer | 27 | 6 | 21 | 0 | 0 | 0% | 0% | 0 |
| Bowl | 17 | 5 | 12 | 4 | 0 | 50% | 0% | 2 |
| Wine | 28 | 28 | 0 | 0 | 0 | 0% | 0% | 4 |

### Phase A conclusion

The Stage-1 Bowl positive does **not** survive. Only 5/30 Bowl candidates have an explicit nominal cause violation, and only two form complete M0/M1/M2 pairs. Their cause-violation counts are M0/M1/M2 = 0/1/1, so neither delayed replanning nor local repair beats immediate full replanning on safety. The small M1 rework observation (41.5 versus 60 median new actions) has worse safety and cannot carry the hypothesis.

## Phase B — replay reconstruction repair

The repaired initializer uses a fresh independent environment, restores the demonstration phase anchor, replays the exact frozen nominal prefix, injects the recorded perturbation at `d`, and continues to every requested branch point. It never shares a mutated branch environment between operators or repetitions.

| Initializer | Insertion pass | All branch-point pass | Contact/outcome agreement | Maximum final-state error |
| --- | ---: | ---: | ---: | ---: |
| Published Stage-1 snapshot restore | 26/30 (86.7%) | not recorded | not recorded | not recorded |
| Snapshot-restore control, five repeats | 26/30 (86.7%) | 147/180 (81.7%) | 149/180 (82.8%) | 3.7161693483379317 |
| Fresh-env prefix reconstruction, five repeats | 30/30 (100%) | 180/180 (100%) | 180/180 (100%) | 0 |

The preregistered replay gate (at least 99%, contact/outcome 100%, error at most `1e-9`) passes. Exact Stage-1 action-chunk hashes match for 30/30 candidates.

Simulation and replay are CPU-side. One local A800 was used only for frozen MLP forward passes because Stage-1 action bytes were created with PyTorch 2.5.1+cu124 CUDA and CPU floating-point bytes differ. No training occurred.

### Phase B conclusion

The Bowl replay problem was caused by the snapshot initializer, not irreducible simulator nondeterminism. Prefix reconstruction repairs it.

## Phase C — policy-free expert action chunks

Phase C uses only demonstration actions `actions[t:t+16]`; no BC policy or policy checkpoint participates. Calibration uses demo IDs 0–9 only. Evaluation IDs 10–39 and held-out IDs 40–49 were never inspected.

### Tasks and the one bounded replacement

- Bowl remains the primary task. The target plate is shifted; the held bowl is never teleported.
- Drawer uses a pre-contact obstacle placement grid.
- Wine-rack is inapplicable: the rack is static model geometry with no movable target joint, while teleporting the manipulated bottle is forbidden.
- The single allowed bounded replacement is LIBERO-GOAL task 6, `put_the_cream_cheese_in_the_bowl`, selected before calibration as the first unused suite-order task with a stable grasp, late release, and movable target joint.

Two preregistration amendments were recorded before the formal grid, based only on demos 0–1 injection-phase smoke and before inspecting any recovery gain: move target-shift timing earlier to avoid existing target contact, and choose the fixed-magnitude x direction away from the held object. Both amendments and their evidence boundary are in `preregistration.yaml`.

### Fixed-grid calibration

- 44 configurations: Bowl 16, Drawer 12, replacement 16.
- 440 nominal calibration records.
- 59 cause-positive candidates received full prefix-stride-1 audits.
- 885/885 independent branch-point reconstructions matched exactly.
- Every prefix branch performs a hard reset, restores the demonstration anchor, replays the exact expert prefix, and then runs an operator in its own environment.

The qualification contract is: injection-instant violation at most 10%, delayed nominal violation 30–80%, actual median recoverable window 2–8, and replay at least 99%. Violation-onset delay is never substituted for a recovery window.

| Task | Qualified configurations | Frozen selection | Selected delayed violation | Median window | Median retention |
| --- | ---: | --- | ---: | ---: | ---: |
| Bowl | 0 | none | n/a | n/a | n/a |
| Drawer | 0 | none | n/a | n/a | n/a |
| Cream cheese in bowl | 3 | `lead08_shift040mm` | 60% | 7 | 50% |

The best Drawer near miss, `offset+00_clear035mm`, has 0% immediate violation and 30% delayed violation, but its actual median recoverable window is 0. Bowl has no configuration satisfying the full phase, immediate-contact, rate, and actual-window contract across all ten calibration demos.

### M0/M1/M2 on the selected replacement configuration

These are calibration-only paired results (`n=6`), not held-out algorithm performance.

| Method | Definition | Cause violations | Safe successes | Median new non-nominal actions |
| --- | --- | ---: | ---: | ---: |
| M0 | immediate full replan at `d` | 1/6 | 5/6 | 34 |
| M1 | continue to `k_last_safe`, then same full replan | 1/6 | 5/6 | 33.5 |
| M2 | continue to same `k_last_safe`, then local repair | 3/6 | 2/6 | 7 |

The timing comparison is nonempty but shows no aggregate safety advantage: M0 and M1 tie on cause violations and safe success, while median new actions differ by only 0.5. M2 uses far fewer new actions but is substantially less safe, so there is no operator win. An oracle best-of-three succeeds safely on 6/6, but that is a privileged calibration observation and does not repair the failed cross-task gate.

### Phase C stopping decision

Only one task has any qualified perturbation. The expert gate requires at least two tasks with median window at least 2 and retention at least 30%. Because calibration parameters must be frozen before evaluation, no result on the single qualified task can create a second qualified task. The failure is therefore monotonic: running evaluation or retuning failed tasks on evaluation data would violate the contract.

Phase C stops with **`KILL_CORE_HYPOTHESIS`**.

## Phases D and E

- Phase D policy substrate: **not executed**. No PAI training job, MLP retraining, Transformer/ACT model, or policy inference was launched.
- Phase E revised policy oracle: **not executed**. No policy-seed oracle records or bootstrap CIs exist.
- Learned/deployable evidence: **none**.

## Answers to the six scientific questions

1. **Q1 — Does the Stage-1 positive survive corrected metrics?** No. Bowl window and post-detection retention both become zero.
2. **Q2 — Can Bowl replay below 99% be repaired?** Yes. Fresh-env prefix reconstruction reaches 180/180 exact branch points with zero final-state error.
3. **Q3 — Is there a real detection → nonzero prefix → no-return window on expert chunks?** On one replacement calibration task, yes; across the required two tasks, no.
4. **Q4 — Does waiting until the last safe prefix reduce rework versus immediate full replan?** Not meaningfully in the only selected calibration cohort: median new actions are 33.5 versus 34, with identical aggregate safety.
5. **Q5 — Is same-trigger local repair better than full replanning?** No. M2 reduces action count but worsens safety to 3/6 cause violations and 2/6 safe successes.
6. **Q6 — Does a stronger small chunk policy reproduce the phenomenon?** Not tested; the Phase C stop rule prohibits Phase D.

## Deliverables and reproduction

Key commands:

```bash
./experiments/r16_p14_libero_stage1b/commands.sh phase-a

R16P14_PYTHON=/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/libero_sft/bin/python \
R16P14_REPLAY_DEVICE=cuda \
./experiments/r16_p14_libero_stage1b/commands.sh phase-b

R16P14_PYTHON=/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/libero_sft/bin/python \
./experiments/r16_p14_libero_stage1b/commands.sh phase-c-calibrate

R16P14_PYTHON=/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/libero_sft/bin/python \
./experiments/r16_p14_libero_stage1b/commands.sh phase-c-summarize

R16P14_PYTHON=/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/libero_sft/bin/python \
./experiments/r16_p14_libero_stage1b/commands.sh test

sha256sum -c artifacts/stage1b/SHA256SUMS
```

Principal outputs:

- Phase A: `artifacts/stage1b/offline_reanalysis/`.
- Phase B: `artifacts/stage1b/replay_gate/branch_reconstructions.jsonl` and summary/report.
- Phase C: `artifacts/stage1b/expert_chunk_calibration/calibration_records.jsonl`, grid summaries, frozen parameters, expert-gate summary, paired CSV, and negative report.
- Contracts: `preregistration.yaml`, `metric_contract.md`, `source_manifest.json`, `decision.json`, and `commands.sh`.
- Tests: `artifacts/stage1b/test_results/pytest.xml`.

The Step-2 Feishu report is: <https://icnbwz7kd1ui.feishu.cn/wiki/UAuJwG7nIixRUMkLWVEcfDyrn1g>.

## Evidence interpretation

- **Metric correction:** negative for the old Bowl claim.
- **Replay correctness:** repaired and passed.
- **Expert mechanism feasibility:** single-task calibration signal only; cross-task gate failed.
- **Policy substrate adequacy:** unknown, because the substrate phase was correctly not reached.
- **Oracle mechanism result:** no revised policy oracle was run.
- **Learned/deployable evidence:** absent by design.
