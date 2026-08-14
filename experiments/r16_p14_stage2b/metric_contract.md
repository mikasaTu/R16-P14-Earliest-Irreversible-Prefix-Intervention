# Stage-2B metric contract

This file freezes operational definitions before formal Stage-2B outcomes are
read. The latest user instruction overrides early stopping only; it does not
relax any scientific gate or permit a positive label after an upstream block.

## Prefix indexing and budgets

- A 16-action actor chunk is indexed `A[0]..A[15]`.
- `d=2` means execute exactly `A[0]` and `A[1]`, then inject.
- Prefix `k` means exactly `A[0:k]` has been executed before replanning.
- `k=d` is immediate replan. `k=H_valid` executes the full validated prefix.
- Every branch receives `B_post=min(200, task_horizon-anchor_step-d)`. Old
  actions retained after detection consume the same budget as new actions.

## Actor action disagreement

For cached action `a_old` and fresh first action `a_fresh`, disagreement is the
root mean square of `(a_old-a_fresh)/actor_action_std` over all seven action
dimensions. The actor's frozen training normalization is used, with its
existing minimum standard deviation. The B6 threshold is the 95th percentile
over clean Phase-A intermediate records only; evaluation outcomes are not read.

## Late-phase and faithfulness

- Stable lift requires the manipulated object's height to exceed its frozen
  task lift threshold for three consecutive executed steps.
- A plausible late placement chunk contains a closed-to-open gripper intent
  within the evaluated horizon and starts within the preregistered target/goal
  distance bound.
- A prefix is faithful when the object is not dropped before a valid target
  release, no wrong release or task-specific catastrophic contact occurs, and
  task progress does not regress by more than 5 cm.
- Task progress is reduction in manipulated-object distance to the registered
  target (XY for cream->bowl; XYZ to the frozen demonstration goal for
  bowl->stove).
- Object drop means stable lift was achieved and height later falls more than
  2 cm while the task is not successful. Wrong release is a gripper opening
  with target distance above the task tolerance. Phase regression is a
  progress decrease greater than 5 cm after late phase was reached.

## Cause safety

- Cream task: after injection, the first closed-to-open gripper transition
  following stable lift is a cause violation when cream-target XY error is
  above the frozen tolerance.
- Stove task: any post-injection contact between the manipulated bowl and the
  registered plate blocker is a cause violation.
- Cause violation is absorbing. Timeout and ordinary failure are never cause
  violations.
- `S(k)=1` iff no target cause violation has occurred before the replan call.
- `safe_success=task_success AND no cause violation`.

## Baselines and oracle

- B0/B1/B2/B3/B4 select `d`, `d+1`, `d+2`, `d+4`, `d+8` when available.
- B5 is the first strict/non-strict local minimum of cached translational speed
  after `d`; a gripper phase boundary is the fallback, then `H_valid`.
- B6 is the first safe prefix whose clean-frozen disagreement threshold is
  exceeded; if none exists it selects `H_valid`.
- `k_last_safe=max{k:S(k)=1}`.
- Within the physically safe set `S(k)=1`, `k_best` maximizes the exact tuple `(safe_success, -new_non_nominal_actions,
  -policy_calls, -k)`. This implements the specified deterministic lowest-index
  final tie-break without adding an unregistered criterion.

## Statistical unit

The event is the independent unit. Prefix branches from one event are paired
measurements and are never resampled as independent rows. Primary confidence
intervals use 10,000 paired event bootstrap resamples with fixed seed 260814,
with task-macro, task, actor-seed and severity-stratified readouts.

## Gate-failure continuation

All downstream runs are completed even if a gate fails. When `H_valid` is
missing or below `d`, descriptive downstream execution uses horizon 8. Such
results are marked `upstream_gate_override=true`; they cannot receive Track A,
Track B, or operator positive labels.

If perturbation qualification is blocked, two distinct diagnostic severities
are frozen without reading any replan/method outcome. Ranking uses replay rate,
the immediate-cause gate, delayed-rate distance to 0.55, first-offset gate,
membership in the Stage-2A frozen pair, then lower severity. This exists only
to execute the user's required downstream matrix and cannot create a signal.

## Secondary operator implementations

The operator audit uses the same event budget as the primary branch and runs
only after all R(k) branches are complete. `full_replan` calls ACT immediately.
`hold_one_step_replan` executes the next cached action (or a neutral pose hold
at the end of the validated chunk) and then replans. `bounded_rollback_replan`
executes one clipped half-scale inverse of the last translational/rotational
action while preserving the closed gripper command, then replans.
`cause_specific_local_repair` uses one deterministic geometry-only action:
cream moves toward the shifted bowl in XY; bowl-stove moves laterally away from
the registered blocker with a small upward component. It reads simulator
geometry but no perturbation type/severity metadata. Every non-ACT operator
action consumes one action from the identical post-detection budget; no
operator receives an extra ACT call. Cause violations remain absorbing.
