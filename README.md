# R16-P14: Earliest-Irreversible-Prefix Intervention

Reproducible Stage-1 oracle-prefix feasibility audit on **LIBERO-GOAL**. This
repository contains the complete implementation, preregistration, tests, PAI
launch contract, raw rollout/oracle records, training logs, final full-state
checkpoints, and the experimental report.

> **Latest Stage-2C decision: `BLOCKED_BY_REPLAY_CONTRACT`.** The 2×A800
> formal job completed all 96/96 event files (5,760 matched-prefix rows plus
> 17,280 recovery rows, zero execution errors), but 33 events produced
> contradictory pre-operator `S_obs(k)` values when ordered branches reused a
> mutable LIBERO runtime. All planned arms and statistics were still run and
> are preserved as diagnostics; no positive, causal, learned, or deployable
> evidence is claimed.

> **Stage-1b decision: `KILL_CORE_HYPOTHESIS`.** Correcting the metric removes
> the Stage-1 Bowl signal. Prefix reconstruction repairs replay to 100%, but a
> policy-free expert-chunk calibration produces a qualified window on only one
> task; the preregistered gate requires two. No new policy or risk model was
> trained.

## Main result

### Stage-2C recoverability/compute-matched result

| Evidence layer | Result |
| --- | --- |
| Contract repair before formal run | PASS |
| Perturbation-family qualification | BLOCKED |
| Formal execution completeness | 96/96 events; 23,040 rows; 0 errors |
| Formal replay contract | BLOCKED; 33/96 events inconsistent |
| Track A / Track B | INCONCLUSIVE / INCONCLUSIVE |
| Accepted / novelty | false / at most N2 |

The complete Chinese report is in
[`experiments/r16_p14_stage2c/reports/REPORT.md`](experiments/r16_p14_stage2c/reports/REPORT.md).
Raw PAI output, the fail-closed decision, every arm, 10,000-draw clustered
statistics, and the code-first reset/order diagnostic are under
[`artifacts/stage2c`](artifacts/stage2c).

```bash
PYTHONPATH="$PWD/experiments/r16_p14_stage2c:$PWD/experiments/r16_p14_stage2b:$PWD/experiments/r16_p14_stage2a" \
  python scripts/verify_r16p14_stage2c.py
```

### Stage-1b contract-repair result

| Evidence layer | Result |
| --- | --- |
| Metric correction | Bowl median window 4 → 0; post-detection retention 50% → 0% |
| Replay correctness | fresh-env reconstruction 180/180 branch points; old snapshot control 147/180 |
| Expert mechanism | 1/3 tasks qualified in calibration; gate requires at least 2 |
| Policy substrate / revised oracle | not executed by the Phase C stopping rule |
| Learned/deployable evidence | none |

The complete Stage-1b report is in
[`experiments/r16_p14_libero_stage1b/reports/REPORT.md`](experiments/r16_p14_libero_stage1b/reports/REPORT.md),
with raw outputs under [`artifacts/stage1b`](artifacts/stage1b).

### Original Stage-1 result

| Task | Clean success | Usable oracle chunks | Median window | Safe-prefix retention | Unsafe reduction | Replay |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Open middle drawer | 10/120 | 27/30 | 0 | 0% | 3.7% | 100% |
| Bowl on plate | 6/120 | 17/30 | 4 | 50% | 64.7% | 86.7% |
| Wine bottle on rack | 11/120 | 28/30 | 0 | 0% | 14.3% | 100% |

Across tasks, unsafe outcomes fell from 72 to 56 (22.2%), median relative
rework reduction was 4.1%, and only one task had a median intervention window
of at least two actions. The detailed interpretation is in
[the experiment report](docs/EXPERIMENT_REPORT.md).

## What is included

- [`experiments/r16_p14_libero_stage1b`](experiments/r16_p14_libero_stage1b):
  corrected metric contract, prefix-reconstruction replay gate, policy-free
  expert-chunk calibration, tests, exact commands, and final stopping decision.
- [`artifacts/stage1b`](artifacts/stage1b): all corrected records, 30×5 Bowl
  replay evidence, 44-config expert calibration records, paired metrics, test
  output, and a SHA-256 manifest.

- [`experiments/r16_p14_libero_stage1`](experiments/r16_p14_libero_stage1):
  chunked state-BC training, deterministic snapshot replay, controlled
  perturbations, five policy-side intervention operators, physical-recovery
  proxy, aggregation, preregistration, and tests.
- [`artifacts/formal_pilot`](artifacts/formal_pilot): all 360 clean rollouts,
  all 90 oracle candidates, schedules, parity records, logs, summaries, and
  completion markers from the successful PAI run.
- [`artifacts/checkpoints`](artifacts/checkpoints): nine final step-8000
  full-state checkpoints, including model, optimizer, scheduler, RNG,
  normalization, and SHA-256 completeness markers.
- [`infra/pai`](infra/pai): sanitized formal PAI launcher/template and exact
  success/recovery provenance. No credential value is committed.
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md): environment, dataset
  hashes, commands, and verification procedure.
- [`scripts/verify_r16p14_release.py`](scripts/verify_r16p14_release.py):
  independent integrity and count validator for this release.
- [`artifacts/SHA256SUMS`](artifacts/SHA256SUMS): SHA-256 manifest covering
  every committed checkpoint, formal result, log, raw record, and test result.

## Quick verification

```bash
python scripts/verify_r16p14_release.py
sha256sum -c artifacts/SHA256SUMS

PYTHONPATH="$PWD:$PWD/experiments/r16_p14_libero_stage1" \
LIBERO_CONFIG_PATH="$PWD/experiments/r16_p14_libero_stage1/libero_config" \
python -m pytest experiments/r16_p14_libero_stage1/tests -q
```

The formal PAI job was `dlc1l9akne34qq7k` on 2×A800 and finished with
`Succeeded`. The frozen experiment source was commit
`a1b61194a8382f5b1a247b9cd9b140645ff2aeb8`, tree
`53001c43fbbb165c0a1f2c71f9cbd4c81b9d0ced`.

## Evidence boundary

This is a bounded Stage-1 pilot: 10 clean episodes per policy seed and
execution horizon, rather than the final preregistered 50. Oracle evaluation
uses policy seed 7; clean evaluation covers seeds 7, 17, and 29. The policy is
a small state-observation BC model—not a VLA, learned risk head, or world
model. Physical recoverability is a privileged scripted proxy, not an
exhaustive proof of the dynamics.

The corresponding Feishu report is available
[here](https://icnbwz7kd1ui.feishu.cn/wiki/DOVIwBUrZi4RAskJW6CcJpOLnif).

The embedded LIBERO source retains its upstream license and provenance; the
original upstream README is preserved at
[`docs/UPSTREAM_LIBERO_README.md`](docs/UPSTREAM_LIBERO_README.md).
