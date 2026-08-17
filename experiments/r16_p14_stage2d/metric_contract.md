# Stage-2D metric and evidence contract

This stage is a fresh-process test of event-aligned cached-prefix reuse. The
universal Stage-1b hypothesis remains `KILLED_IMMUTABLE`; Stage-2C remains
blocked by its replay contract. Nothing in Stage-2D can reverse either result.

The independent unit is an init-state/rollout event. Prefixes, methods,
severities, and ACT checkpoints are repeated measurements. Confidence
intervals use a paired event bootstrap with 10,000 resamples and seed 216214.
Prefix rows are never pooled as independent samples.

`S_obs(k)=1` exactly when the preregistered target-cause violation has not
occurred before prefix `k`. The label is absorbing. A `0 -> 1` transition
invalidates the event and is counted as a gate failure; it is never repaired
or silently dropped.

`safe_success = task_success AND NOT cause_violation`. The target-shift cause
is a closed-to-open gripper transition after stable lift while the cream
cheese lies outside the shifted bowl region. The path-obstacle cause is
post-injection contact between the manipulated bowl and the registered
blocker. The manipulated object is never teleported.

Every formal `(event, k, arm, repeat)` is reconstructed in a unique process
created by `multiprocessing.get_context("spawn")`. No environment, simulator,
controller, wrapper, observable cache, RNG, history, or branch context is
shared. Missing rows and exceptions are failures.

The pre-tail signature covers simulator state, observation feature, four-state
history, three-action history, robot qpos/qvel, contact pairs, manipulated and
target/obstacle qpos, gripper state, task phase, CauseTracker state, and exact
executed-prefix action bytes. `CACHED_MATCHED` and `CACHED_NOQUERY` must have
identical signatures and `S_obs(k)` because the former actor call is discarded
and actor inference must have no environment/history side effect.

The primary causal H2 contrast is `EVENT_ALIGNED_CACHED -
FRESH_MATCHED_AT_RULE_K` at the same `k`, with equal prefix length, required
pre-`k` calls, tail horizon four, maximum action budget, maximum actor-call
budget, and episode horizon. Calls are measured, never padded after success.
Cached-action count is explanatory only and cannot establish efficiency.

The H3 contrast is `EVENT_ALIGNED_CACHED - IMMEDIATE_FRESH`. H4 compares the
frozen event rule against the strongest fixed delay chosen on calibration
only. The evaluation split cannot be opened until the frozen-rule files are
committed and hashed. An evaluation all-`k` oracle is appendix-only and cannot
change a primary decision.

Scientific gates are evaluated exactly as preregistered. Because the user
explicitly required all planned experiments even if a gate fails, downstream
work after a failure is clearly stamped `diagnostic_only=true`; it cannot
produce a positive, accepted, N3, or N4 label.
