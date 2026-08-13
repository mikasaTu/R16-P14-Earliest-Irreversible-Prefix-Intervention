# Stage-1b metric contract

This contract is frozen before any Stage-1b simulator run. It corrects the
semantic conflation found in the Stage-1 aggregation code.

## Outcome labels

For every method branch, labels are separate and may overlap:

- `cause_violation`: the task-specific monitor emitted an explicit registered
  cause violation (`branch.violation == true`). Budget exhaustion alone is
  never a cause violation.
- `task_failure`: the task was not completed within the branch outcome
  (`branch.task_recoverable == false`).
- `timeout`: no explicit cause violation occurred, the task was incomplete,
  and the equal branch budget was exhausted.
- `safe_success`: the task completed without an explicit cause violation.

The primary corrected Stage-1 cohort contains candidates whose no-intervention
branch at detection has `cause_violation == true`. Results for task-failure-only
candidates are reported separately and never silently added to safety counts.

## Timing and retention

- `d` is the registered perturbation insertion/detection prefix.
- `k_last_safe` is the largest tested prefix at or after `d` for which at least
  one deployable intervention (`trim`, `hold`, bounded rollback, or
  cause-specific repair) has `safe_success == true`.
- If an explicit-cause candidate has no deployable successful intervention,
  its window and retention are zero. `null` is reserved for candidates outside
  the explicit-cause primary cohort.
- `intervention_window = max(0, k_last_safe - d)`.
- `retention_after_detection = (k_last_safe - d) / (H - d)`, where `H` is
  chunk length. This replaces the Stage-1 `k_last_safe / H` metric.

## Paired methods

All comparisons use the same candidate and branch budget:

- M0 `immediate_full_replan_at_d`: Stage-1 `trim_and_replan` at `d`.
- M1 `continue_to_last_safe_then_full_replan`: retain the original actions
  from `d` through `k_last_safe`, then `trim_and_replan` at that prefix.
- M2 `continue_to_last_safe_then_cause_specific_repair`: retain the same
  original actions through the same `k_last_safe`, then run the registered
  local repair.

Signed deltas follow the requested algebra exactly:

- `timing_gain = M1 - M0`
- `operator_gain = M2 - M1`
- `total_gain = M2 - M0`

For cost/violation metrics, a negative signed delta is beneficial. For safe
success, a positive signed delta is beneficial. Positive-valued `*_savings`
fields are also emitted for readability.

## Corrected rework

`new_non_nominal_actions_after_detection` counts newly executed actions after
the intervention point that are not part of the retained original nominal-safe
prefix. Therefore:

- M0 rework is its branch `steps` from `d`.
- M1/M2 rework is branch `steps` from `k_last_safe`; the retained `k-d`
  nominal actions are not rework.
- `discarded_nominal_actions` is `H-d` for M0 and `H-k_last_safe` for M1/M2.
- `completion_steps_after_detection` includes retained nominal actions plus
  post-intervention branch steps.

Stage-1 records contain branch path length but not the per-action original
chunk required to reconstruct retained-prefix path length offline. Phase A
therefore reports `branch_path_length` and explicitly marks corrected total
path length unavailable; Phase B+ records the needed actions and reports full
post-detection path length.

## Missingness and denominators

Paired M0/M1/M2 metrics require an explicit-cause candidate with a defined
`k_last_safe`. Every table reports this paired sample size. No failed or
undefined pair is converted into a zero-valued gain.
