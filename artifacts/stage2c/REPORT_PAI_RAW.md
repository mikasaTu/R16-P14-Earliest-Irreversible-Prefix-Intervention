# R16-P14 Stage-2C — Recoverability-Defined and Compute-Matched Validation

Overall: **BLOCKED_BY_REPLAY_CONTRACT**. Track A: **INCONCLUSIVE**; Track B: **INCONCLUSIVE**. `accepted=false`; novelty remains `N2_ORACLE_PROTOCOL_BOUNDARY_ONLY`.

## Immutable negative evidence

Stage-1b's universal hypothesis remains `KILLED_IMMUTABLE`. Stage2B remains blocked on actor-conditioned perturbation and actor-history replay; its Track A/B results remain inconclusive. Nothing in this stage overwrites or reinterprets those records.

## Correctness repairs

The sole historical replay failure first becomes observable at global step 181: a 1.86e-09 history delta is amplified into a 6.47e-05 ACT chunk delta. Three new reconstructions are mutually byte-exact. Stage2C therefore uses outcome-blind 3/3 admission and excludes unstable events; it does not relax tolerances.

Replay rates now divide by all attempts and require zero errors. Prefix safety is absorbing, and goal progress comes from the live BDDL predicate/site/object geometry rather than demo-0.

## Qualified perturbation families

Qualification status: **BLOCKED**. Target-shift two-severity gate: `False`; second family: `put_the_bowl_on_the_stove` with gate `False`. All later experiments ran even when a gate failed.

Actor events: 238 admitted from 240 completed attempts; unstable exclusion rate 0.000. Formal minimum-data gate: `True`.

## Compute-matched cached-prefix value

Track A evaluation rows: 39. Safe-success difference (cached minus fresh): 0.183 with cluster-bootstrap 95% CI [0.05, 0.3333333333333333]. Mean new-action reduction: -6.943.

Both primary branches execute exactly `k-d` prefix actions, make one detection-time call and one call at k, then use h=16 with identical action budgets. `CACHED_NOQUERY` remains secondary deployment evidence.

## Operator-relative recoverability

Atlas rows: 1440; event boundaries: 96; invalid/nonmonotonic prefix events: 33. Median recoverable windows by task: `{'put_the_bowl_on_the_stove': 14.0, 'put_the_cream_cheese_in_the_bowl': 13.0}`. `k_irrev_U` is an operator-family persistent crossing, not a physical irreversibility point.

## Cross-fitted replanability

Leave-one-recovery-actor-out rows: 117. Frozen calibration-only baseline: `k_last_recoverable`. Held-out safe-success difference: -0.094 with 95% CI [-0.2333333333333333, 0.03888888888888888]; mean new-action reduction: -1.254.

Held-out outcomes never enter prefix selection. Actor seed is a repeated measurement within `(task, init_state_id)`; all reported intervals use the preregistered 10,000-draw cluster bootstrap.

## Why mechanisms raise or lower performance

Across 1440 matched prefix cells, cached reuse improves safe outcome in 113 and worsens it in 60. The reverse audit attributes gains only to retained nominal progress/fewer new actions under equal compute; losses are checked against stale-prefix cause violations and larger cached/fresh action displacement. This is an explanation of frozen code behavior, not a new idea.

## Retired mechanisms and evidence boundary

Local repair remains `RETIRED_NO_SIGNAL`; the operator router remains `RETIRED_NO_SIGNAL`. No learned predictor, RGB/VLA, world model, π0.5, or deployable policy was trained. Learned/deployable evidence remains **NONE**.

## Execution completeness and decision

Formal matrix: 5760 matched rows, 17280 recovery rows, 0 errors, `all_complete=True`. Upstream gates: `{'contract_repair': True, 'formal_matrix': True, 'minimum_data': True, 'replay_contract': False, 'second_failure_family': False}`.

Final decision: **BLOCKED_BY_REPLAY_CONTRACT**. Raw Track-A/Track-B criteria were `False` / `False`, but positive labels are impossible whenever an upstream gate is false.
