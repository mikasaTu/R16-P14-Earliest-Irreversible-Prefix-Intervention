# Stage-2C code-first mechanism reverse audit

This is a reverse explanation of frozen observations. It introduces no new method or idea.

## Qualification mechanism

All 891 qualification attempts completed with 0 missing attempts, but the gate was `BLOCKED` (`BLOCKED_BY_SECOND_FAILURE_FAMILY`).
For target shift, 3/30 chunks contained a post-detection gripper release, and 0 were eligible after the lift condition. This confirms why all frozen target-shift severities had zero delayed cause violations.
Path-obstacle failures are summarized from every frozen cell in the JSON companion; no task or severity was selected using method outcomes.

## Formal replay-contract mechanism

All planned formal rows completed, but 363/1440 recovery prefix cells produced contradictory pre-operator `S_obs(k)` values, affecting 33/96 events. The replay contract is therefore `BLOCKED`.
The recovery code records the prefix label before recovery-actor inference and before operator actions. Because all recovery rows execute the same cached prefix, actor/operator/tail outcomes cannot explain the disagreement. The remaining code boundary is residual mutable state or branch-order dependence in the shared EventRuntime reset path; the artifacts do not identify one hidden field uniquely.

## Matched-prefix mechanism

The evaluation contains 720 planned cached/fresh prefix pairs. Raw cached outcomes improved safe success in 52 cells and worsened it in 36 cells.
The JSON companion reports action disagreement, retained progress, cause violations, completion steps, paths and contacts separately for improvements and degradations. These are descriptive partitions, not causal pairs, because the replay contract failed.
`new_non_nominal_actions` is partly structural: cached prefix actions are defined as retained, while fresh-prefix and tail actions are defined as new. A reduction without safe-success improvement is not an independent efficiency mechanism.

## Causal scope

The intended cached/fresh contract matches the detection-time call, call at k, h=16 tail, action budget, simulator seed, perturbation and actor. The observed reset/order dependence means a branch difference is not empirically isolated to the k-d prefix actions in this run. Associations with disagreement or progress only describe the frozen code path and do not establish a universal physical irreversibility mechanism.

All downstream outcomes remain diagnostic-only when the perturbation-family gate is blocked. `accepted=false`, Stage-1b remains `KILLED_IMMUTABLE`, and novelty remains at most N2.
