<title>step5</title>

Continue the repository:

[https://github.com/mikasaTu/R16-P14-Earliest-Irreversible-Prefix-Intervention](https://github.com/mikasaTu/R16-P14-Earliest-Irreversible-Prefix-Intervention)

Use current main HEAD as the immutable parent.  
At the start, record the exact HEAD, tree, status, and hashes.

Create:

R16-P14 Stage-2C  
Recoverability-Defined and Compute-Matched Validation

Branch:

agent/stage2c-recoverability-matched-validation

Directories:

experiments/r16_p14_stage2c/  
artifacts/stage2c/

Do not overwrite any Stage-1, Stage-1b, Stage-2A, or Stage-2B  
source or artifact.

# ==================================================  
0. Immutable conclusions

The following conclusions are immutable:

- Stage-1b universal hypothesis = KILLED_IMMUTABLE
- Stage-2B chunk executability = PASS
- Stage-2B H_valid = 16
- Stage-2B actor-conditioned perturbation = BLOCKED
- Stage-2B actor-history replay = BLOCKED
- Stage-2B Track A = INCONCLUSIVE
- Stage-2B Track B = INCONCLUSIVE
- Stage-2B local repair = NO_SIGNAL
- Stage-2B operator router = NO_OPERATOR_ROUTING_SIGNAL
- novelty ceiling = N2_ORACLE_PROTOCOL_BOUNDARY_ONLY
- accepted = false

This stage must not reverse or weaken those conclusions.

No π0.5, RGB/VLA, world model, learned replanability head,  
learned irreversibility head, or operator-router training is allowed.

This is a mechanism-validation stage only.

==================================================

1. Scientific hypotheses  
==================================================

Track A:

Recoverability-Bounded Cached Prefix Reuse.

A cached action prefix has incremental value only if, at an identical  
prefix length and identical compute/execution budget, executing the  
cached prefix is safer or more efficient than executing a freshly  
generated prefix.

Track B:

History-Conditioned Replanability.

The best state for policy re-entry depends on physical state, actor  
history, and remaining cached chunk, and is not necessarily the  
detection state, the last observed-safe state, or the last recoverable  
state.

Do not call any result physical irreversibility.

Use the terms:

- operator-relative recoverability
- policy-family-relative irreversibility
- last observed-safe prefix
- last recoverable prefix
- cross-fitted best replan prefix

# ==================================================  
2. Stage-2C frozen evidence

Freeze and hash:

- current repository HEAD/tree
- Stage-2B final evidence commit/tree
- Stage-2B decision/report
- three ACT checkpoints: seeds 7, 17, 29
- actor config and normalization
- LIBERO source/config
- task init states
- all Stage-2B raw artifacts
- current test suite
- current Python/MuJoCo/PyTorch environment

Create:

experiments/r16_p14_stage2c/  
preregistration.yaml  
metric_contract.md  
commands.sh  
source_freeze/  
contract_repair/  
task_qualification/  
actor_events/  
replay/  
matched_prefix/  
recoverability_atlas/  
crossfit_replanability/  
tests/  
reports/  
decision.json

# ==================================================  
3. Phase A — Correctness and contract repair

A1. Replay denominator

In perturbation qualification:

replay_rate =  
successful reconstructions / all attempted reconstructions

Every exception or missing record counts as replay failure.

Add an explicit gate:

error_count == 0 for every formal severity/task cell.

A2. Prefix-safety monotonicity

For every event, the independently reconstructed prefix labels must obey:

S_obs(k+1) <= S_obs(k)

where S_obs means no target cause violation has occurred before k.

Any 0→1 transition invalidates the whole event and must produce:

NONMONOTONIC_CAUSE_PREFIX

Do not compute k_last_observed_safe for an invalid event.

A3. Goal-region geometry

For stove, plate, drawer, and similar tasks, compute progress and  
target distance to the actual BDDL goal region or object geometry.

Do not use demonstration-0 final object position as the task goal.

A4. Replay root-cause audit

For every Stage-2B replay failure, identify the first divergence in:

- init state
- each pre-anchor action
- simulator state
- observation feature
- four-state history
- three-action history
- original actor chunk

Record:

- first divergent global step
- first divergent tensor/key
- max numerical difference
- controller/wrapper/RNG state at that point

Fix the root cause where possible.

A formal Stage-2C event is admitted only if, before reading any  
intervention outcome:

- 3/3 fresh reconstructions pass
- anchor state/history exact
- original chunk hash exact
- branch order invariant
- no error record

Report unstable-event exclusion rate separately.

A5. Statistics

The independent cluster is:

(task, init_state_id)

Actor seed is a repeated measurement inside that cluster.

Implement hierarchical/cluster paired bootstrap with 10,000 resamples.

Do not treat actor-seed variants or prefix rows as independent samples.

A6. Baseline selection

Any strongest-baseline selection must use calibration data only.

Freeze the baseline name before evaluation outcomes are loaded.

# ==================================================  
4. Phase B — Two qualified failure families

Primary target-shift task:

put_the_cream_cheese_in_the_bowl

Perturbation:

move the bowl after two cached actions.  
Never teleport the manipulated cream cheese.

Frozen calibration grid:

[0.04, 0.06, 0.08] meters

Second failure family must be a future-path obstacle, not another  
ordinary target shift.

Candidate tasks, in frozen order:

1. put_the_bowl_on_the_stove
2. push_the_plate_to_the_front_of_the_stove
3. open_the_top_drawer_and_put_the_bowl_inside

Use the first task that passes qualification.  
Do not inspect Track-A or Track-B gains while selecting it.

For the path-obstacle family:

- derive the manipulated-object future swept path from the actor's  
frozen nominal chunk
- choose future path index from:  
[d+4, d+8, d+12]
- choose lateral clearance from:  
[0.00, 0.02, 0.04] meters
- no contact is allowed at injection
- the manipulated object itself must not be teleported

A task/severity qualifies only if:

- valid calibration events >=20
- at least two actor seeds contribute >=5 events
- immediate cause violation <=0.10
- delayed cause violation in [0.30, 0.80]
- median first-violation offset >=2
- at least 20% of violations occur at an interior prefix,  
not only at the final one or two actions
- replay rate computed over all attempts >=0.99
- error_count ==0
- two distinct severities qualify

Use only fresh, preregistered calibration init IDs.

For current Stage-2B primary tasks, use previously uninspected IDs:

30–39: Stage-2C calibration  
40–49: Stage-2C evaluation

If a backup task is needed, freeze an unused calibration/evaluation  
split before running outcomes.

If no non-target-shift task qualifies, stop with:

BLOCKED_BY_SECOND_FAILURE_FAMILY

Do not continue with two semantically equivalent target-shift tasks.

# ==================================================  
5. Phase C — Actor-generated formal event pool

Every formal event must come from the frozen ACT itself.

No demonstration action chunk may be used as the nominal prefix.

Each event stores:

- task
- generator actor seed/checkpoint
- init-state ID/hash
- pre-anchor actions
- anchor simulator state
- exact four-state history
- exact three-action history
- complete 16-action actor chunk
- all hashes
- contact/task phase
- target-region geometry
- replay-admission evidence

One event per actor rollout.

Evaluation minimum:

- 

  > =24 admitted events per task
- 

  > =6 events per generator actor seed per task
- 

  > =8 events per severity per task

Do not replace missing events with calibration events.

# ==================================================  
6. Phase D — Compute-matched prefix counterfactual

Detection/injection prefix:

d = 2

For every admitted event and every:

k in [d, d+1, ..., 16]

run the following primary pair.

CACHED_MATCHED(k):

1. reconstruct the exact event
2. execute A[0:d]
3. inject perturbation
4. call ACT once at d and save the fresh chunk, but do not execute it  
(matched-compute sham call)
5. execute cached actions A[d:k]
6. at k call ACT
7. complete the task using execution_horizon=16

FRESH_MATCHED(k):

1. reconstruct the exact event
2. execute A[0:d]
3. inject perturbation
4. call ACT once at d
5. execute the fresh chunk's first k-d actions
6. at k call ACT
7. complete the task using execution_horizon=16

The two branches must have identical:

- number of actor calls before k
- open-loop prefix length
- tail execution horizon
- post-detection action budget
- maximum task horizon
- simulator seed
- perturbation
- generator actor
- evaluation actor

Also run as secondary deployment evidence:

CACHED_NOQUERY(k)

This branch does not make the sham call at d and therefore measures  
actual compute savings. It is not the primary causal comparison.

Additional baselines:

- immediate fresh replan
- fixed delays 1, 2, 4, 8
- velocity-phase heuristic
- cached-vs-fresh action disagreement
- full old chunk
- hold-prefix matched control

Do not use h=1 as the primary post-replan execution mode.  
All primary methods use the same h=16 tail policy.

# ==================================================  
7. Phase E — Operator-relative recoverability atlas

Freeze the recovery operator set:

U = {  
fresh_h16,  
fresh_h4,  
hold_one_step_then_fresh_h16,  
rollback_one_step_then_fresh_h16  
}

Use identical action and policy-call budgets for all operators.

At each prefix k, run every recovery operator with the three frozen  
ACT checkpoints 7, 17, and 29.

Record:

C_self(k):  
safe-success of the event's own actor checkpoint

C_family(k):  
safe-success fraction over the three recovery actor checkpoints

Define:

R_U(k)=1

iff at least one operator in U has:

C_family(k) >= 2/3

Define:

k_last_observed_safe =  
max k with no observed pre-replan target cause violation

k_last_recoverable =  
max k with R_U(k)=1

k_irrev_U =  
minimum k such that R_U(j)=0 for every j>=k

If no persistent crossing exists in the chunk:

k_irrev_U = null

Never call k_irrev_U a physical irreversibility point.

Record raw nonmonotonic recovery patterns; do not silently force them  
to be monotonic. The persistent-crossing definition is used only for  
the operator-relative boundary.

# ==================================================  
8. Phase F — Track A evaluation

Track A primary comparison:

CACHED_MATCHED(k_last_recoverable)  
vs  
FRESH_MATCHED(k_last_recoverable)

Also compare the recoverability-boundary policy against immediate  
fresh h16 replanning.

Primary metrics:

- safe success
- target cause violation
- task success
- retained cached actions
- new non-nominal actions
- actor calls
- total actions
- completion steps
- path length
- contact count
- task progress retained at k
- cached-versus-fresh action displacement

Track A signal requires:

1. both tasks median recoverable window >=2
2. cluster-bootstrap 95% CI lower bound of safe-success difference

   > = -0.03
3. new non-nominal actions reduced by >=15%
4. direction consistent in both tasks
5. direction consistent in at least 2/3 actor seeds
6. direction consistent under both severities
7. no result depends on a task that failed perturbation qualification
8. compute-matched and budget-matched contracts pass

Allowed positive label:

OPERATOR_RELATIVE_PREFIX_REUSE_SIGNAL

This does not reverse Stage-1b universal kill.

# ==================================================  
9. Phase G — Cross-fitted Track B evaluation

For every event, perform leave-one-recovery-actor-out selection:

- use two actor checkpoints to estimate the best replan prefix
- evaluate the selected prefix only on the held-out third checkpoint
- rotate all three held-out checkpoints

Selection tuple on the two fitting actors:

1. higher safe-success fraction
2. fewer new non-nominal actions
3. fewer actor calls
4. lower prefix index final tie-break

The held-out actor outcome must never participate in prefix selection.

Freeze the strongest deployable baseline on calibration data only.

Baseline set:

- immediate fresh h16
- fixed delay 1/2/4/8
- velocity-phase heuristic
- action-disagreement trigger
- k_last_observed_safe
- k_last_recoverable

Track B signal requires:

1. held-out safe success improves by >=8 percentage points  
OR safe-success is non-inferior and new actions fall by >=15%
2. cluster-bootstrap CI lower bound >0 for the claimed primary gain
3. selected prefix differs from d in >=20% of events
4. selected prefix differs from k_last_recoverable in >=20%
5. an interior prefix is selected in >=20%
6. positive direction appears in both tasks
7. positive direction appears in at least 2/3 actor seeds
8. no extra action budget or actor-call budget explains the gain

Allowed positive label:

CROSSFIT_REPLANABILITY_SIGNAL

Do not train a predictor in Stage-2C.

# ==================================================  
10. Local repair and operator router

Do not optimize or train local repair.

Preserve the Stage-2B result:

local repair = NO_SIGNAL  
operator router = NO_OPERATOR_ROUTING_SIGNAL

At most, run the existing local repair on a small diagnostic subset  
to confirm compatibility after code repair.

It cannot receive a positive label in Stage-2C.

# ==================================================  
11. Required tests

Add tests for at least:

1. replay errors count in the replay-rate denominator
2. any replay error blocks a formal severity cell
3. S_obs(k) cannot transition 0→1
4. nonmonotonic prefix events are excluded
5. goal-region distance does not depend on demonstration-0 endpoint
6. replay-admission uses no method outcomes
7. used events pass 3/3 reconstruction
8. cached and fresh matched branches execute equal prefix lengths
9. cached and fresh matched branches have equal pre-k actor calls
10. both primary branches use h=16 tail execution
11. old actions consume the same budget as fresh actions
12. CACHED_MATCHED's sham chunk is not executed
13. demonstration chunks are forbidden
14. k_last_observed_safe differs from k_last_recoverable when expected
15. k_irrev_U uses persistent crossing
16. no physical-irreversibility label is emitted
17. held-out actor outcomes cannot enter k selection
18. strongest baseline is frozen from calibration only
19. bootstrap clusters by task/init-state
20. actor-seed rows inside a cluster are not independent
21. task/severity/seed minimum-data gates fail closed
22. second failure family cannot be chosen by method gain
23. completed evidence cannot be silently overwritten
24. interrupted event shards resume only incomplete events
25. no positive label can be issued after an upstream gate failure

Run an integration smoke on one event from each failure family before  
the formal matrix.

# ==================================================  
12. Statistics

Primary unit:

cluster = (task, init_state_id)

Within cluster:

- actor generator seed
- recovery actor seed
- severity
- prefix branches

Use:

- paired hierarchical bootstrap
- 10,000 resamples
- fixed bootstrap seed
- per-task reporting
- per-severity reporting
- per-actor-seed reporting
- macro task average

Do not pool prefix rows as independent observations.

Do not select tasks, severities, baseline names, or thresholds using  
evaluation outcomes.

# ==================================================  
13. Decision schema

decision.json must contain:

stage1b_universal_hypothesis:  
KILLED_IMMUTABLE

stage2b_status:  
BLOCKED_UPSTREAM

contract_repair:  
PASS | BLOCKED

replay_contract:  
PASS | BLOCKED

second_failure_family:  
PASS | BLOCKED

track_a_operator_relative_prefix_reuse:  
SIGNAL | NO_SIGNAL | INCONCLUSIVE | NOT_RUN

track_b_crossfit_replanability:  
SIGNAL | NO_SIGNAL | INCONCLUSIVE | NOT_RUN

local_repair:  
RETIRED_NO_SIGNAL

operator_router:  
RETIRED_NO_SIGNAL

overall:  
PROCEED_TO_LEARNED_REPLANABILITY  
PROCEED_TO_OPERATOR_RELATIVE_BOUNDARY_ONLY  
STOP_R16_P14_PREFIX_TIMING_FAMILY  
BLOCKED_BY_REPLAY_CONTRACT  
BLOCKED_BY_SECOND_FAILURE_FAMILY  
BLOCKED_BY_MINIMUM_DATA  
BLOCKED_BY_INFRA

accepted must remain false.  
Novelty must remain no higher than N2 in this stage.

Decision logic:

- Track A SIGNAL + Track B SIGNAL:  
PROCEED_TO_LEARNED_REPLANABILITY
- Track A NO_SIGNAL + Track B SIGNAL:  
PROCEED_TO_LEARNED_REPLANABILITY  
and explicitly retire the original cached-prefix claim
- Track A SIGNAL + Track B NO_SIGNAL:  
PROCEED_TO_OPERATOR_RELATIVE_BOUNDARY_ONLY
- Track A NO_SIGNAL + Track B NO_SIGNAL:  
STOP_R16_P14_PREFIX_TIMING_FAMILY

# ==================================================  
14. Resource limits

- use the existing frozen ACT checkpoints
- no actor retraining
- maximum 2 GPUs
- maximum 2 concurrent GPU workers
- CPU-first simulation and aggregation
- no π0.5
- no world model
- no learned head
- no large hyperparameter sweep
- no HTML
- no formal activation infrastructure

Stop after Stage-2C report even if a positive signal is obtained.

# ==================================================  
15. Checkpoint and deliverables

After Phase A and Phase B, create and commit a checkpoint containing:

1. exact HEAD/tree
2. all repaired contracts
3. replay-failure root-cause report
4. old vs corrected replay-rate calculation
5. S(k) monotonicity audit
6. goal-region implementation audit
7. qualified task/failure-family roster
8. frozen severities
9. admitted-event counts
10. exact next command

Automatically continue only if all required gates pass.

Final deliverables:

experiments/r16_p14_stage2c/  
preregistration.yaml  
metric_contract.md  
commands.sh  
source_freeze/  
contract_repair/  
task_qualification/  
actor_events/  
replay/  
matched_prefix/  
recoverability_atlas/  
crossfit_replanability/  
tests/  
reports/  
decision.json

artifacts/stage2c/  
raw attempts  
admitted actor events  
replay traces  
cached/fresh matched branches  
recovery-operator branches  
cross-fit records  
hierarchical bootstrap results  
negative results  
source and artifact hashes  
SHA256SUMS

Final REPORT.md must clearly separate:

- immutable Stage-1b negative result
- Stage-2B descriptive evidence
- correctness repairs
- qualified perturbation families
- compute-matched cached-prefix value
- operator-relative recoverability
- cross-fitted replanability
- local-repair retirement
- learned/deployable evidence, which remains NONE
- novelty status, which remains no higher than N2
