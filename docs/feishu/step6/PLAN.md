继续仓库：

[https://github.com/mikasaTu/R16-P14-Earliest-Irreversible-Prefix-Intervention](https://github.com/mikasaTu/R16-P14-Earliest-Irreversible-Prefix-Intervention)

使用当前 main HEAD 作为不可变父提交。

创建新阶段：

R16-P14 Stage-2D  
Fresh-Process Event-Aligned Prefix Reuse

创建 branch/worktree：

agent/stage2d-fresh-process-event-aligned-prefix-reuse

创建目录：

experiments/r16_p14_stage2d/  
artifacts/stage2d/

不得覆盖、修改或删除：

experiments/r16_p14_libero_stage1/  
experiments/r16_p14_libero_stage1b/  
experiments/r16_p14_stage2a/  
experiments/r16_p14_stage2b/  
experiments/r16_p14_stage2c/

artifacts/formal_pilot/  
artifacts/stage1b/  
artifacts/stage2a/  
artifacts/stage2b/  
artifacts/stage2c/

# ==================================================  
0. Immutable conclusions

以下结论不可修改、弱化、翻转或重新解释：

- Stage-1b universal hypothesis = KILLED_IMMUTABLE
- Stage-2B chunk executability = PASS
- Stage-2B H_valid = 16
- Stage-2B actor-conditioned perturbation = BLOCKED
- Stage-2B actor-history replay = BLOCKED
- Stage-2B Track A = INCONCLUSIVE
- Stage-2B Track B = INCONCLUSIVE
- Stage-2B local repair = NO_SIGNAL
- Stage-2B operator router = NO_OPERATOR_ROUTING_SIGNAL
- Stage-2C qualification = BLOCKED
- Stage-2C replay contract = BLOCKED
- Stage-2C Track A = INCONCLUSIVE
- Stage-2C Track B = INCONCLUSIVE
- accepted = false
- novelty ceiling = N2_ORACLE_PROTOCOL_BOUNDARY_ONLY

Stage-2D 不验证 physical irreversibility。  
只允许使用以下术语：

- observed-safe prefix
- policy-relative recoverability
- last recoverable prefix
- event-aligned handoff
- cached-prefix reuse
- oracle upper bound

禁止使用：

- physical irreversibility point
- universally irreversible prefix
- validated VLA method
- accepted idea
- N3/N4

本轮禁止：

- π0.5
- RGB/VLA
- world model
- learned risk/replanability head
- operator router training
- actor retraining
- hyperparameter sweep
- evaluation-set retuning
- shared mutable runtime across branches
- formal activation/WORM/大型 provenance infrastructure
- HTML

只使用普通 Git、JSONL、CSV、Markdown、NPZ 和 SHA256。

==================================================

# 1. Scientific hypotheses

Stage-2D 独立验证以下假设：

H1 — Nontrivial observed-safe window

After perturbation detection at d, at least two cached actions remain free  
of the explicitly preregistered target-cause violation in two different  
failure families.

H2 — Cached-prefix content value

At the same prefix length k, with matched detection-time actor calls,  
the same tail actor, same tail horizon, same action budget and same  
environment seed, executing the old cached prefix A[d:k] is safer or  
more efficient than executing an equally long freshly generated prefix.

H3 — Event-aligned handoff value

An outcome-blind event-aligned handoff rule is safety-noninferior to  
immediate replanning and reduces at least one real execution cost.

H4 — Nontrivial interior selection

The best or event-aligned handoff point is not almost always d or H.

The original universal R16-P14 hypothesis remains killed regardless  
of Stage-2D result.

# ==================================================  
2. Repository and evidence freeze

At the start, record:

- exact current HEAD/tree/status
- Stage-1b final decision/report hashes
- Stage-2C decision/report/replay-diagnostic hashes
- ACT checkpoints seed 7/17/29
- actor config and normalization hashes
- LIBERO source/config hashes
- Python/PyTorch/MuJoCo/robosuite versions
- current test suite hash
- GPU/CPU environment
- source paths and artifact manifests

Create:

experiments/r16_p14_stage2d/  
preregistration.yaml  
metric_contract.md  
commands.sh  
source_freeze/  
branch_isolation/  
init_pool/  
actor_events/  
perturbation_qualification/  
calibration_atlas/  
frozen_rule/  
confirmatory_evaluation/  
oracle_appendix/  
statistics/  
tests/  
reports/  
decision.json

artifacts/stage2d/  
source_freeze/  
branch_isolation/  
init_pool/  
actor_events/  
perturbation_qualification/  
calibration_atlas/  
frozen_rule/  
confirmatory_evaluation/  
oracle_appendix/  
statistics/  
test_results/  
SHA256SUMS

# ==================================================  
3. Phase A — Fresh-process branch isolation

The Stage-2C EventRuntime/reset implementation must not be used for  
formal Stage-2D branches.

Every independent:

(event, prefix_k, arm, repeat)

must execute in a fresh spawned process:

multiprocessing.get_context("spawn")

Required reconstruction:

fresh process  
→ create fresh LIBERO env  
→ load frozen init state  
→ replay all pre-anchor actor-executed actions  
→ reconstruct exact 4-state history  
→ reconstruct exact 3-action history  
→ call the frozen ACT and verify the event chunk hash  
→ execute cached A[0:d]  
→ inject perturbation  
→ verify detection signature  
→ execute branch-specific prefix  
→ verify pre-tail signature  
→ execute the common tail  
→ close process

Never reuse:

- environment
- wrapper
- controller
- observable cache
- simulator
- RNG object
- branch context

between arms or prefix positions.

Use deterministic ACT inference. Prefer CPU inference for Stage-2D.  
If CPU chunk bytes differ from old GPU artifacts, regenerate and freeze  
new Stage-2D actor events. Do not modify old artifacts and do not require  
old Stage-2A/2B chunk hashes.

Pre-tail signature must include:

- simulator state/hash
- observation feature/hash
- four-state history/hash
- three-action history/hash
- robot qpos/qvel/hash
- contact pairs/hash
- manipulated object qpos/hash
- target/obstacle qpos/hash
- gripper state
- task phase/hash
- CauseTracker state/hash
- exact executed-prefix action hash

Same-action controls:

CACHED_MATCHED:  
call actor at d, discard output, execute A[d:k]

CACHED_NOQUERY:  
do not call actor at d, execute the identical A[d:k]

Their pre-tail signatures and S_obs(k) must be exactly equal.

Run the isolation gate on:

- both target-shift and path-obstacle smoke events
- k = d, d+4, H
- three independent repeats
- at least three arm-order permutations

Gate:

- 100% reconstruction
- 100% same-action signature match
- 100% contact/cause agreement
- no 0→1 S_obs transition
- no branch-order dependence
- no missing/error records
- max state error <=1e-9
- actor inference has no environment/history side effect

Failure:

BLOCKED_BY_BRANCH_ISOLATION

Stop immediately. Do not run perturbation qualification or any atlas.

Output:

branch_isolation/raw.jsonl  
branch_isolation/signatures.jsonl  
branch_isolation/summary.json  
branch_isolation/report.md

# ==================================================  
4. Phase B — Fresh init-state pool

Do not reuse old 0–49 IDs as a new held-out set.

Generate and freeze 100 fresh valid initial states per selected task,  
using deterministic reset/randomization seeds and checking only:

- initial BDDL predicates
- collision validity
- object/workspace validity
- simulator reset validity

Do not inspect intervention or method outcomes.

Split before any outcome:

0–9:  
infrastructure only

10–39:  
event and perturbation calibration

40–79:  
confirmatory evaluation

80–99:  
untouched reserve; Stage-2D must not read

Write:

init_pool/init_states.npz  
init_pool/manifest.json  
init_pool/splits.json  
init_pool/report.md

Hash every state and the complete pool.

# ==================================================  
5. Phase C — Actor-generated event construction

Use frozen HistoryConditionedStateACT checkpoints:

7, 17, 29

No retraining and no tuning.

All formal chunks must be generated by the actor.  
Demonstration chunks are permitted only for geometry/debug smoke and  
must never enter formal mechanism evidence.

Remove every global-step fallback event rule.

Family A:

task:  
put_the_cream_cheese_in_the_bowl

Event eligibility:

- stable grasp
- stable lift for at least two steps
- current gripper closed
- task not already successful
- generated chunk has a close-to-open transition
- release index r satisfies 6 <= r <= 13
- nominal prefix has measurable transport progress
- remaining chunk length is 16
- at least three actions between d and release
- no global-step fallback

Set d=2.

Family B:

task:  
put_the_bowl_on_the_stove

Event eligibility:

- stable grasp and lift
- no current bowl/blocker contact
- task not already successful
- nominal object displacement >=0.06 m
- future path point exists at d+4, d+7 or d+10
- future point is >=0.05 m from current bowl position
- at least three actions before predicted future-path event
- no global-step fallback

Each event stores:

- task
- actor checkpoint/seed/hash
- init-state ID/hash
- pre-anchor actions/hash
- anchor global step
- anchor state/hash
- four-state history/hash
- three-action history/hash
- 16-step actor chunk/hash
- gripper transition index
- nominal object trajectory
- object/target/obstacle geometry
- task phase
- contacts
- source_is_actor_generated_chunk=true
- source_is_demonstration_chunk=false

One event per rollout.

Required calibration availability before perturbation tuning:

- 

  > =40 eligible events/task
- 

  > =10 events from each actor seed/task
- 

  > =30 distinct init-state IDs/task

Otherwise:

BLOCKED_BY_EVENT_CONSTRUCTION

# ==================================================  
6. Phase D — Event-aligned perturbation qualification

Qualification may inspect only:

- event eligibility
- phase validity
- injection contact
- immediate cause violation
- delayed old-chunk cause violation
- first violation index
- replay validity

It must not inspect:

- replan success
- cached success
- fresh-prefix success
- k_best
- k_last_recoverable
- cost reduction
- Track A/B gain

Family A perturbation:

move the target bowl only.  
Never move or teleport the manipulated cream cheese.

Severities:

0.04, 0.06, 0.08 m

Direction must be lateral to the nominal object-to-target approach,  
not a fixed world-axis sign.

Cause:

after stable lift, the gripper transitions from closed to open while  
the manipulated object is outside the preregistered new target region.

Cause violation is absorbing as a label.

Family B perturbation:

place the existing blocker at a future swept-path point.  
Never teleport the manipulated bowl.

Future indices:

d+4, d+7, d+10

Placement:

p_obs = p_future + lateral_unit \*  
(object_radius + obstacle_radius + clearance_delta)

clearance_delta:

-0.010, 0.000, +0.010 m

Object and obstacle radii must come from live object geometry/AABB or a  
documented bounding-sphere approximation.

No contact is permitted at injection.

Cause:

post-injection contact between the manipulated bowl and the registered  
blocker.

A task/severity qualifies only if:

- valid calibration events >=20
- at least two actor seeds each contribute >=5
- injection contact rate =0
- immediate cause violation <=0.05
- delayed old-chunk cause violation in [0.30, 0.80]
- median first-violation offset >=3
- at least 30% of violations occur at an interior prefix  
d+3 <= onset <= H-2
- replay =100%
- error_count=0
- nonmonotonic S_obs event count=0

Each family requires two distinct qualified severities.

No forced diagnostic continuation is allowed.

If only one family qualifies:

SINGLE_FAMILY_ONLY

Stop before confirmatory evaluation.

If neither qualifies:

STOP_R16_P14_PREFIX_TIMING_FAMILY

Output:

perturbation_qualification/raw_attempts.jsonl  
perturbation_qualification/prefix_safety.jsonl  
perturbation_qualification/grid_summary.csv  
perturbation_qualification/frozen_parameters.json  
perturbation_qualification/negative_results.md  
perturbation_qualification/report.md

# ==================================================  
7. Mandatory checkpoint barrier

Before any calibration atlas, commit and report:

1. exact HEAD/tree
2. files changed
3. branch-isolation implementation
4. replay results
5. fresh init-pool hashes and splits
6. eligible event counts by task/seed
7. perturbation qualification table
8. frozen two severities per family
9. exact branch volume
10. current blocker
11. exact next command

Do not start the atlas unless:

- branch isolation PASS
- both families have two qualifying severities
- event minimum counts PASS

Automatically continue only if all gates pass.  
Do not ask the user for confirmation.

# ==================================================  
8. Phase E — Calibration oracle atlas

Use calibration split only.

For every calibration event and:

k = d, d+1, ..., H

run each branch in a separate fresh spawned process.

Primary arms:

B0_IMMEDIATE_FRESH:

- execute A[0:d]
- inject perturbation
- call ACT at d
- complete with common tail

CACHED_MATCHED(k):

- execute A[0:d]
- inject perturbation
- call ACT at d and discard output
- execute A[d:k]
- call ACT at k
- complete with common tail

FRESH_MATCHED(k):

- execute A[0:d]
- inject perturbation
- call ACT at d
- execute fresh_d[0:k-d]
- call ACT at k
- complete with common tail

HOLD_MATCHED(k):

- execute A[0:d]
- inject perturbation
- call ACT at d
- execute k-d hold actions
- call ACT at k
- complete with common tail

Secondary:

CACHED_NOQUERY(k)  
FULL_OLD_CHUNK

Primary tail execution horizon:

h_tail=4

All primary arms use the same:

- recovery actor checkpoint
- environment/init seed
- perturbation
- prefix length
- tail horizon
- maximum action budget
- maximum policy-call budget
- maximum episode horizon

Do not pad successful branches with dummy tail actor calls.  
Actual actor calls and inference time are outcomes.

Record:

- S_obs(k)
- pre-tail signature
- cause violation
- task success
- safe success
- actual post-detection actions
- completion steps
- actual actor calls
- inference wall time
- total branch wall time
- EEF path length
- manipulated-object path length
- task progress at k
- progress regression
- contact count
- cached actions retained
- cached/fresh action disagreement
- gripper/contact event indices
- all budgets and hashes

Define oracle k only after all branches exist:

lexicographic objective:

1. safe success
2. no cause violation
3. fewer actual post-detection actions
4. shorter completion time
5. fewer actual actor calls
6. lower k as final deterministic tie-break

The oracle is an upper bound, not a deployable algorithm.

Oracle mechanism gate requires:

- both tasks median observed-safe window >=2
- 

  > =30% events have an interior safe/recoverable prefix
- k_oracle != d in >=30% events
- k_oracle != H in >=30% events
- oracle improves safe success by >=10 percentage points over the  
strongest fixed baseline, OR is safety-noninferior and reduces a real  
execution cost by >=15%
- signal appears in both tasks and both severities

Failure:

STOP_R16_P14_PREFIX_TIMING_FAMILY

Do not train a selector.

# ==================================================  
9. Phase F — Freeze outcome-blind event-aligned rule

Using calibration data only, freeze:

Target shift:

k_rule = clip(release_index - 2, d, H)

Path obstacle:

k_rule = clip(predicted_path_intersection_index - 2, d, H)

Freeze from calibration only:

- strongest fixed delay
- action-disagreement threshold
- cost definitions
- safety tolerance epsilon=0.03
- bootstrap seed
- all evaluation methods
- tail horizon
- maximum budgets

Write:

frozen_rule/rule.json  
frozen_rule/calibration_selection.json  
frozen_rule/baselines.json  
frozen_rule/report.md

No evaluation outcome may be loaded before these files are committed  
and hashed.

# ==================================================  
10. Phase G — Confirmatory evaluation

Use evaluation split 40–79 only.

Do not scan all k before applying the frozen rule.

Methods:

1. IMMEDIATE_FRESH
2. FIXED_DELAY_2
3. FIXED_DELAY_4
4. FIXED_DELAY_8
5. EVENT_ALIGNED_CACHED
6. FRESH_MATCHED_AT_RULE_K
7. HOLD_MATCHED_AT_RULE_K
8. CACHED_NOQUERY_AT_RULE_K
9. FULL_OLD_CHUNK

Minimum data per task:

- 

  > =30 valid independent evaluation events
- 

  > =12 events per severity
- 

  > =8 events per actor seed
- 

  > =24 distinct init-state clusters
- every event replay 3/3 exact
- one event maximum per rollout

Primary comparisons:

H2:  
EVENT_ALIGNED_CACHED  
vs  
FRESH_MATCHED_AT_RULE_K

H3:  
EVENT_ALIGNED_CACHED  
vs  
IMMEDIATE_FRESH

H4:  
EVENT_ALIGNED_CACHED  
vs  
strongest frozen fixed baseline

After all primary evaluation and decision files are frozen, an  
evaluation oracle atlas may be run as an appendix only.  
It must not affect the primary decision.

# ==================================================  
11. Metric contract

Primary safety metrics:

- explicit target cause violation
- task success
- safe success
- paired safe-success difference
- paired cause-violation difference

Primary efficiency metrics:

- actual post-detection actions
- completion steps
- actual actor calls
- actual inference wall time
- EEF path length
- manipulated-object path length
- branch wall time
- progress retained/regressed

Do not use new_non_nominal_actions as a primary metric.

Cached-action count is only an explanatory metric and cannot itself  
establish efficiency.

H1 PASS:

- both tasks median safe window >=2
- 

  > =30% events have at least two interior safe cached actions
- both severities agree

H2 PASS:

- paired safe-success CI lower bound >=-0.03
- cause violation increase <=0.03
- at least one real efficiency metric improves >=15%
- both tasks agree
- both severities agree
- at least 2/3 actor seeds agree

H3 PASS:

- EVENT_ALIGNED_CACHED is safety-noninferior to IMMEDIATE_FRESH
- at least one real efficiency metric improves >=15%
- result is not driven by one task, severity or actor seed
- equal budgets pass

H4 PASS:

- rule selects an interior prefix in >=20% events
- differs from strongest fixed delay in >=20%
- recovers >=40% of oracle efficiency gap
- safety-noninferior to strongest fixed baseline

# ==================================================  
12. Statistics

The independent event unit is the init-state/rollout event.

Prefix branches are not independent samples.  
Actor checkpoints are repeated model measurements and must be reported  
separately.

Use:

- paired event bootstrap
- 10,000 resamples
- fixed bootstrap seed
- per-task confidence intervals
- per-severity reporting
- per-actor-seed reporting
- macro task average as descriptive only
- exact paired count tables for binary safe success
- no pooling of prefix rows as independent observations

Because there are only two task families, do not claim population-wide  
cross-task generalization from a task bootstrap.

# ==================================================  
13. Required tests

Add real and unit tests covering at least:

1. every formal branch gets a unique spawned process
2. no environment/runtime object is shared across branches
3. same-action cached controls have identical pre-tail signatures
4. actor inference does not mutate env/history
5. branch-order permutation leaves signatures and outcomes unchanged
6. reconstructing the same branch three times is exact
7. any replay error blocks the event
8. any S_obs 0→1 transition blocks the event
9. global-step fallback events are forbidden
10. target-shift events require a future release transition
11. path-obstacle events require a future swept-path point
12. manipulated object cannot be teleported
13. injection contact blocks the qualification cell
14. perturbation selection reads no method outcome
15. calibration/evaluation/reserve splits are disjoint
16. evaluation cannot load before frozen_rule hashes exist
17. CACHED and FRESH execute equal prefix lengths
18. CACHED and FRESH have equal required pre-k calls
19. all primary tails use h=4
20. all arms have equal maximum action budgets
21. actual calls are measured and not padded after success
22. cached-action reuse is not counted as efficiency by itself
23. oracle evaluation cannot affect the primary decision
24. bootstrap resamples events, not prefix rows
25. no positive label after any upstream failure
26. no physical-irreversibility wording is emitted
27. no demonstration chunk enters formal evidence
28. incomplete event shards resume without overwriting completed evidence
29. checksum manifest covers all Stage-2D artifacts
30. a two-task real LIBERO integration smoke passes before formal launch

# ==================================================  
14. Decision schema

decision.json must contain:

stage1b_universal_hypothesis:  
KILLED_IMMUTABLE

stage2c_status:  
BLOCKED_UPSTREAM_IMMUTABLE

branch_isolation:  
PASS | BLOCKED

event_construction:  
PASS | BLOCKED

target_shift_qualification:  
PASS | BLOCKED

path_obstacle_qualification:  
PASS | BLOCKED

oracle_mechanism:  
PASS | NO_ORACLE_GAP | NOT_RUN

h1_observed_safe_window:  
PASS | FAIL | INCONCLUSIVE | NOT_RUN

h2_cached_prefix_content:  
PASS | FAIL | INCONCLUSIVE | NOT_RUN

h3_event_aligned_handoff:  
PASS | FAIL | INCONCLUSIVE | NOT_RUN

h4_nontrivial_selection:  
PASS | FAIL | INCONCLUSIVE | NOT_RUN

cached_prefix_claim:  
SUPPORTED_CONDITIONALLY  
NO_CACHED_PREFIX_CONTENT_VALUE  
SINGLE_FAMILY_ONLY  
RETIRED

overall:  
PROCEED_TO_LEARNED_REPLANABILITY  
PROCEED_REPLAN_TIMING_ONLY  
LOCAL_MECHANISM_ONLY_NOT_DEPLOYABLE  
SINGLE_FAMILY_SIGNAL_ONLY  
STOP_ALL_PREFIX_TIMING  
BLOCKED_BY_BRANCH_ISOLATION  
BLOCKED_BY_EVENT_CONSTRUCTION  
BLOCKED_BY_PERTURBATION_QUALIFICATION  
BLOCKED_BY_MINIMUM_DATA  
BLOCKED_BY_INFRA

accepted:  
false

novelty:  
N2_ORACLE_PROTOCOL_BOUNDARY_ONLY

Decision logic:

- H1 FAIL:  
STOP_ALL_PREFIX_TIMING
- H1 PASS, H2 FAIL:  
cached_prefix_claim=NO_CACHED_PREFIX_CONTENT_VALUE  
If H3/H4 pass:  
PROCEED_REPLAN_TIMING_ONLY  
Else:  
STOP_ALL_PREFIX_TIMING
- H1 PASS, H2 PASS, but H3 or H4 fail:  
LOCAL_MECHANISM_ONLY_NOT_DEPLOYABLE
- H1/H2/H3/H4 all pass:  
PROCEED_TO_LEARNED_REPLANABILITY
- only one task family passes:  
SINGLE_FAMILY_SIGNAL_ONLY

Never set accepted=true.

# ==================================================  
15. Resource rules

- reuse existing frozen ACT checkpoints
- no actor retraining
- CPU-first simulation and actor inference
- maximum 2 GPUs
- maximum 2 concurrent GPU workers
- local integration smoke before any PAI job
- no external PAI job unless local execution is insufficient
- no large sweep
- no forced downstream continuation after a scientific gate fails

If a gate fails, stop at that gate and write the full negative report.  
Do not spend compute on a known-invalid formal matrix.

# ==================================================  
16. Deliverables

Required:

- preregistration.yaml
- metric_contract.md
- source freeze and hashes
- fresh init-state pool
- branch-isolation raw evidence
- same-action purity control
- actor event records
- perturbation qualification records
- negative results
- calibration atlas
- frozen non-oracle rule
- confirmatory paired rows
- oracle appendix
- paired statistics and confidence intervals
- tests and integration smoke
- decision.json
- REPORT.md
- exact commands
- SHA256SUMS
- exact final Git commit/tree

Final REPORT.md must clearly separate:

- immutable Stage-1b negative conclusion
- Stage-2C contaminated diagnostic evidence
- branch-isolation correctness
- event opportunity validity
- perturbation qualification
- oracle mechanism upper bound
- cached-vs-fresh causal comparison
- event-aligned vs immediate comparison
- confirmatory versus oracle evidence
- learned/deployable evidence, which is NONE in Stage-2D
- novelty, which remains no higher than N2

Do not claim:

- general R16-P14 validation
- physical irreversibility
- VLA validation
- π0.5 result
- learned intervention policy
- N3/N4
- accepted idea
