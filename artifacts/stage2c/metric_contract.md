# Stage-2C metric and evidence contract

The independent cluster is `(task, init_state_id)`. Generator actor seed,
recovery actor seed, severity, prefix, and branch are repeated measurements
inside that cluster. All paired intervals use a 10,000-draw hierarchical
cluster bootstrap with frozen seed `160214`.

`replay_rate` is successful reconstruction attempts divided by **all**
attempted reconstructions. Exceptions and missing records are failures. A
formal task/severity cell additionally requires `error_count == 0`.

`S_obs(k)` is one exactly when no target-cause violation has occurred before
prefix `k`. It is absorbing: any `0 -> 1` transition invalidates the entire
event as `NONMONOTONIC_CAUSE_PREFIX`; no last-observed-safe index is then
reported.

Goal progress is measured against the live BDDL goal predicate geometry. An
`on(object, object)` goal uses the target object's live body geometry and the
LIBERO predicate's 3 cm horizontal support radius. An `on(object, site)` or
`in(object, site)` goal uses the live site pose, rotation, size, and the same
point-in-region convention as LIBERO. Demo-0 endpoints are prohibited.

`safe_success = task_success AND NOT target_cause_violation`. Cause labels are
absorbing, but every branch receives the same post-detection action budget and
maximum horizon even after a cause label is observed.

The primary causal contrast is `CACHED_MATCHED(k) - FRESH_MATCHED(k)`. Both
branches make one actor call at detection, execute exactly `k-d` prefix
actions, make one actor call at `k`, and use `execution_horizon=16` thereafter.
The cached branch's detection-time call is a sham and its returned actions are
never executed. `CACHED_NOQUERY` is secondary deployment evidence only.

Recoverability is relative to the frozen operator and ACT-checkpoint family.
`k_irrev_U` is a persistent operator-family crossing, never a physical
irreversibility claim. Evaluation outcomes cannot change tasks, severities,
baseline identity, thresholds, or admission.

