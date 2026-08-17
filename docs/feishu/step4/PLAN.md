<title>step4</title>

继续仓库：

[https://github.com/mikasaTu/R16-P14-Earliest-Irreversible-Prefix-Intervention](https://github.com/mikasaTu/R16-P14-Earliest-Irreversible-Prefix-Intervention)

当前不可变事实：

- Stage-1b universal hypothesis = KILLED_IMMUTABLE
- Stage-2A checkpoint = PASS_READY_FOR_BOUNDED_ATLAS
- Stage-2A full atlas was not launched
- Track A = INCONCLUSIVE
- Track B = INCONCLUSIVE
- accepted = false
- current Policy-Relative Replanability novelty boundary =  
N2_ORACLE_PROTOCOL_BOUNDARY_ONLY

禁止删除、覆盖、弱化或重新解释上述结论。

创建新阶段：

R16-P14 Stage-2B  
Action-Chunk Faithfulness and Actor-Conditioned Replanability Pilot

本轮目标：

1. 验证当前 ACT 的 action-chunk suffix 是否真实可执行；
2. 在 ACT 自己生成的 chunk 上重新校准 delayed invalidation；
3. 验证完整 actor history 的 branch reconstruction；
4. 运行 bounded prefix-replanability Atlas pilot；
5. 分开评价 conditional original Track A 和 replanability Track B。

本轮禁止：

- π0.5
- RGB/VLA
- world model
- learned irreversibility head
- learned replanability head
- operator router training
- novelty 升级
- accepted=true
- HTML
- formal activation / WORM / 大型 provenance infrastructure

只使用普通 Git、JSONL、CSV、Markdown、SHA256。  
最多使用 2 张 GPU；CPU-first。  
不得覆盖 Stage-1、Stage-1b、Stage-2A artifacts。

创建：

branch:  
agent/stage2b-actor-conditioned-atlas-pilot

experiments/r16_p14_stage2b/  
preregistration.yaml  
metric_contract.md  
commands.sh  
chunk_executability/  
actor_events/  
perturbation_qualification/  
replay/  
atlas_pilot/  
operator_audit/  
tests/  
reports/  
decision.json

artifacts/stage2b/

# ==================================================  
0. Freeze current evidence

冻结并记录：

- current repository HEAD/tree
- Stage-1 commit/tree
- Stage-1b commit/tree
- Stage-2A evidence commit/tree
- Stage-2A decision/report hashes
- ACT checkpoints seed 7/17/29
- actor model/config/normalization hashes
- LIBERO config/source hashes
- task-init-state hashes
- selected Stage-2A perturbation hashes

decision.json 必须始终保留：

stage1b_universal_hypothesis: KILLED_IMMUTABLE  
accepted: false

==================================================

1. Phase A — Action-chunk executability audit  
==================================================

使用现有冻结 HistoryConditionedStateACT：

- seed 7
- seed 17
- seed 29

Primary tasks:

- put_the_cream_cheese_in_the_bowl
- put_the_bowl_on_the_stove

不得重新训练或调参。

测试：

execution_horizon in [1, 2, 4, 8, 16]

含义：

生成 16-step chunk，  
连续执行前 execution_horizon 个动作，  
再重新调用 ACT。

每个：

task × actor_seed × horizon

至少运行 20 个不同 init states。  
不得重复同一个 init-state hash 凑数。

记录：

- clean success
- grasp/lift/late-phase reach
- object drop
- wrong release
- phase regression
- policy calls
- episode length
- path length
- chunk smoothness
- original chunk hash
- at each intermediate step:  
old chunk next action  
fresh actor first action  
normalized disagreement

同时构建 late-phase prefix-faithfulness audit：

对于进入 late phase 的 anchor，  
执行同一 chunk 的前 h 步，  
记录：

- prefix remains phase-valid
- grasp remains valid
- no task-specific catastrophic event
- task progress delta
- fresh-action disagreement curve

定义 H_valid 为满足以下条件的最大 h：

1. both-task aggregate clean success >= h1 success - 0.15
2. both-task late-phase reach >= h1 late-phase reach - 0.15
3. late-phase prefix-faithful rate >=0.80
4. at least 2/3 actor seeds have success drop <=0.20
5. no simulator or action-contract error

需要：

H_valid >= 8

否则停止为：

BLOCKED_BY_CHUNK_EXECUTABILITY

该结果只说明 actor substrate 不适合 action-chunk prefix experiment，  
不得解释为 R16-P14 mechanism failure。

输出：

chunk_executability/raw_episodes.jsonl  
chunk_executability/prefix_records.jsonl  
chunk_executability/summary.json  
chunk_executability/report.md

# ==================================================  
2. Phase B — Build actor-generated event pool

正式 event 必须来自 ACT 自己的 clean closed-loop rollout，  
不能从 demonstration action chunk 直接生成。

每个 event：

1. 从 frozen init state 启动；
2. ACT 使用 execution_horizon=1 运行至目标 late phase；
3. ACT 真实生成一个 chunk A；
4. 保存完整 actor and simulator context。

保存：

- task
- actor seed/checkpoint
- init-state ID/hash
- pre-anchor executed action sequence
- anchor global step
- simulator state/hash
- last 4 observations/features/hashes
- last 3 executed actions/hashes
- task ID
- complete 16-step chunk
- chunk hash
- phase/contact/gripper/object state

每个 episode 最多一个 event。

首先确认每任务至少 50 个不同 init states。  
如果不足，停止为：

BLOCKED_BY_EVENT_SPLIT

固定 split：

0–9:  
actor qualification only, not reused

10–19:  
actor-conditioned perturbation calibration

20–29:  
Stage-2B Atlas pilot evaluation

30–49:  
reserved, not inspected in Stage-2B

这些 split 只能称：

perturbation-selection splits

不得称 policy-held-out。

Event eligibility:

cream-cheese → bowl:

- object stably lifted
- gripper currently closed
- actor chunk has a plausible late placement/release phase
- at least H_valid actions available
- task not already successful

bowl → stove:

- bowl stably lifted
- actor chunk nominal replay approaches target region later
- no current blocker contact
- at least H_valid actions available
- task not already successful

# ==================================================  
3. Phase C — Actor-conditioned perturbation qualification

Detection/injection prefix:

d = 2

即执行 a0、a1 后注入环境变化。  
本轮授予 oracle detection，但 actor 不得读取 perturbation type、  
severity 或 hidden metadata。

所有 qualification 必须沿 ACT 自己的 frozen chunk 进行。

Task 1:  
put_the_cream_cheese_in_the_bowl

Perturbation:  
move bowl, never move manipulated cream cheese.

先测试 Stage-2A frozen magnitudes。  
若不合格，只允许一次 preregistered grid：

[0.03, 0.04, 0.05, 0.06] meters

Task 2:  
put_the_bowl_on_the_stove

Perturbation:  
place existing scene object in the actor chunk's nominal future target region.  
Never teleport manipulated bowl.

先测试 Stage-2A frozen offsets。  
若不合格，只允许一次 preregistered grid：

[0.04, 0.05, 0.06, 0.075] meters

选择配置只能使用 calibration init IDs 10–19，并且只能看：

- phase validity
- immediate target-cause violation
- delayed nominal target-cause violation
- first violation offset
- replay validity

禁止查看：

- immediate replan success
- last-safe success
- k_best
- rework reduction
- proposed method gain

Cause labels:

cream-cheese task:

misaligned release after target shift:  
gripper opens after stable lift and object-target XY error exceeds  
the preregistered tolerance.

bowl-stove task:

post-injection contact between manipulated bowl and registered blocker.

Cause violation is absorbing.

Do not count timeout or ordinary task failure as cause violation.

Each task requires two distinct qualifying severities.

Qualification gates:

- at least 20 valid calibration events/task across seeds
- at least 2 actor seeds contribute >=5 events/task
- immediate cause violation <=0.10
- delayed nominal cause violation in [0.30, 0.80]
- median first-violation offset after d >=2
- replay rate >=0.99
- two distinct severities qualify

Both tasks must qualify.

Otherwise stop as:

BLOCKED_BY_ACTOR_CONDITIONED_PERTURBATION

输出：

perturbation_qualification/raw.jsonl  
perturbation_qualification/grid_summary.csv  
perturbation_qualification/frozen_parameters.json  
perturbation_qualification/negative_results.md  
perturbation_qualification/report.md

# ==================================================  
4. Phase D — Full actor-history replay gate

Use:

2 tasks  
× 2 severities  
× 3 actor seeds  
× 3 calibration events  
× 4 prefix positions  
× 3 repeats

Prefix positions:

d  
min(d+2, H_valid)  
middle prefix  
H_valid

Every reconstruction:

fresh env  
→ restore exact init state  
→ replay all pre-anchor actor-executed actions  
→ verify anchor simulator state  
→ reconstruct exact 4-state history  
→ reconstruct exact 3-action history  
→ call actor and verify original chunk hash  
→ execute first d actions  
→ inject same perturbation  
→ execute to prefix k  
→ verify branch state/history/cause tracker  
→ call actor and verify replanned chunk hash

No shared mutable env between branches.

Exact checks:

- simulator state/hash
- observations/features/hashes
- state-history sequence
- action-history sequence
- contact pairs/hash
- object and gripper state
- task phase
- original action-chunk hash
- replanned action-chunk hash
- cause tracker state

Gate:

- state/history/chunk reconstruction >=0.99
- contact/outcome agreement =1.0
- no branch-order dependence
- max numerical state error <=1e-9 where exact state is supported

Failure:

BLOCKED_BY_ACTOR_HISTORY_REPLAY

输出：

replay/raw_reconstructions.jsonl  
replay/summary.json  
replay/report.md

# ==================================================  
5. Automatic continuation rule

After Phases A–D, write a checkpoint report.

If and only if:

- H_valid >=8
- both tasks have two qualifying severities
- actor-history replay PASS

then automatically continue to Phase E.

Do not wait for user confirmation.

Stop only on a failed gate or after Phase E/F completion.

# ==================================================  
6. Phase E — Bounded Safe-Replanability Atlas pilot

Use evaluation init IDs 20–29 only.

Maximum event pool:

2 tasks  
× 3 actor seeds  
× 10 init states  
= 60 events

Alternate the two frozen severities within each task.

Minimum valid data:

- 

  > =20 valid events/task
- 

  > =8 events/severity/task
- 

  > =5 events/actor-seed/task

For each event:

A = frozen actor-generated chunk  
effective horizon = H_valid  
d = 2

Evaluate every prefix length:

k in [d, d+1, ..., H_valid]

Index definition:

k is the number of original chunk actions already executed before replanning.

k=d:  
immediate replan

k=H_valid:  
execute the complete validated old chunk before replan

Branch R(k):

1. reconstruct exact actor event
2. execute first d actions
3. inject perturbation
4. execute original actions A[d:k]
5. if target cause violation occurred:  
record S(k)=0 and do not call this a safe prefix
6. otherwise:  
record S(k)=1  
call the same frozen ACT at s_k  
complete the task with execution_horizon=1

Use the same post-detection action budget for all k in an event:

B_post = min(200, remaining task horizon after detection)

Actions spent continuing the old chunk consume B_post.

Record:

- pre-replan cause violation
- task success
- safe success
- timeout
- old nominal actions retained after detection
- nominal actions discarded
- new non-nominal actions
- policy calls
- total actions
- completion steps
- path length
- contact events
- object drop
- task phase
- original/replan chunk hashes
- wall time

# ==================================================  
7. Baselines derived from the same branches

N0:  
full old chunk, k=H_valid

B0:  
immediate replan, k=d

B1:  
delay 1 action

B2:  
delay 2 actions

B3:  
delay 4 actions

B4:  
delay 8 actions, if valid

B5:  
velocity-phase heuristic:  
first registered local speed minimum / phase boundary after d

Do not call this an official PACE reproduction.

B6:  
cached-versus-fresh action disagreement

At each k compare:

old chunk's next action  
vs  
fresh actor's first action

Freeze threshold from clean calibration events only,  
using the 95th percentile of normalized disagreement.

A_original:  
k_last_safe = max{k: S(k)=1}

N_oracle:  
k_best selected lexicographically by:

1. safe success
2. fewer new non-nominal actions
3. fewer policy calls
4. deterministic lowest index final tie-break

N_oracle is an oracle protocol result, not a deployable algorithm.

# ==================================================  
8. Track A evaluation

Track A tests only the conditional late-target-invalidation family.

Primary comparison:

A_original vs B0

Report per task, severity, actor seed and macro average:

- median k_last_safe-d
- safe success
- cause violation
- retained nominal actions
- new non-nominal actions
- policy calls
- completion steps
- path length

Track-A pilot signal requires:

1. both tasks median k_last_safe-d >=2
2. paired 95% CI lower bound of safe-success difference >= -0.05
3. new non-nominal actions reduction >=15%
4. direction consistent on at least 2/3 actor seeds
5. same direction under both severities
6. result not driven by only one task or one seed

Allowed positive label:

CONDITIONAL_LATE_PLACEMENT_PILOT_SIGNAL

It must not reverse Stage-1b's universal kill.

# ==================================================  
9. Track B evaluation

Compare N_oracle with the strongest of:

B0–B6 and A_original

Track-B oracle pilot signal requires:

1. safe success +10 percentage points  
OR cause violation relative reduction >=25%
2. k_best != d in >=20% events
3. k_best != k_last_safe in >=20% events
4. d < k_best < k_last_safe in >=15% events
5. effect appears in both tasks
6. effect is not explained by larger post-detection budget
7. all policy-call differences are explicitly reported

Allowed positive label:

REPLANABILITY_ORACLE_PILOT_SIGNAL

Do not train a diagnostic probe in Stage-2B.

# ==================================================  
10. Secondary operator audit

Only after the primary R(k) atlas completes.

At:

d  
k_best  
k_last_safe

compare:

- full replan
- hold one step + replan
- bounded rollback + replan
- existing cause-specific local repair

Use identical post-detection budgets.

Operator routing signal requires:

- at least two operators each uniquely safe-win on >=10% events
- unique winners occur in both tasks
- no operator receives extra action or policy-call budget

Otherwise:

operator_router = NO_OPERATOR_ROUTING_SIGNAL

Do not train a router.

# ==================================================  
11. Statistics

Unit of independence:

event, not prefix branch

Use:

- paired event bootstrap
- task-stratified bootstrap
- actor-seed-stratified reporting
- severity-stratified reporting
- 10,000 resamples
- fixed bootstrap seed
- 95% confidence intervals

Do not pool all prefix rows as independent samples.

Report:

- per-task
- per-severity
- per-actor-seed
- macro task average

# ==================================================  
12. Required code and tests

Add modules equivalent to:

- chunk_executability.py
- actor_event_builder.py
- actor_perturbation_qualification.py
- actor_history_replay.py
- atlas_runner.py
- atlas_aggregate.py
- baselines.py
- bootstrap.py

Required unit/metamorphic tests:

1. prefix indexing has no off-by-one error
2. k=d exactly equals immediate-replan baseline
3. k=H_valid executes the full validated old prefix
4. cause violation is absorbing
5. timeout is not cause violation
6. k_last_safe is computed correctly
7. k_best lexicographic tie-break is deterministic
8. branch order permutation does not change outcomes
9. changing actor checkpoint causes reconstruction rejection
10. changing state history causes reconstruction rejection
11. changing action history causes reconstruction rejection
12. changing original chunk bytes causes rejection
13. formal Atlas never uses demonstration chunk as nominal prefix
14. event splits are disjoint
15. calibration never reads evaluation/held-out outcomes
16. interrupted Atlas resumes only incomplete events
17. non-empty completed evidence is never silently overwritten
18. event budget is identical across prefix methods
19. all frozen method outputs are derived from the same R(k) branches
20. primary statistics resample events, not branch rows

Run an integration smoke on one event before full bounded pilot.

# ==================================================  
13. Decision schema

decision.json must contain:

stage1b_universal_hypothesis:  
KILLED_IMMUTABLE

stage2a_checkpoint:  
PASS_READY_FOR_BOUNDED_ATLAS

chunk_executability:  
PASS | BLOCKED

H_valid:  
integer_or_null

actor_conditioned_perturbation:  
PASS | BLOCKED

actor_history_replay:  
PASS | BLOCKED

track_a_conditional_timing:  
PILOT_SIGNAL | NO_SIGNAL | INCONCLUSIVE | NOT_RUN

track_a_local_repair:  
PILOT_SIGNAL | NO_SIGNAL | NOT_RUN

track_b_replanability:  
PILOT_SIGNAL | NO_ORACLE_GAP | INCONCLUSIVE | NOT_RUN

operator_router:  
PILOT_SIGNAL | NO_OPERATOR_ROUTING_SIGNAL | NOT_RUN

overall:  
PROCEED_TO_FULL_ATLAS_WITH_BOTH_TRACKS  
PROCEED_TO_FULL_ATLAS_REPLANABILITY_ONLY  
PROCEED_TO_FULL_ATLAS_CONDITIONAL_TIMING_ONLY  
STOP_PREFIX_TIMING_FAMILY  
BLOCKED_BY_CHUNK_EXECUTABILITY  
BLOCKED_BY_ACTOR_CONDITIONED_PERTURBATION  
BLOCKED_BY_ACTOR_HISTORY_REPLAY  
BLOCKED_BY_EVENT_SPLIT  
BLOCKED_BY_INFRA

accepted must remain false.

# ==================================================  
14. Resource and execution rules

- maximum 2 GPUs
- maximum 2 concurrent GPU workers
- CPU first for simulation/replay
- no external PAI job unless local execution is impossible
- no π0.5
- no world model
- no learned probe/head
- no hyperparameter sweep
- no HTML
- no formal activation infrastructure

The bounded pilot is expected to be approximately:

<= 60 events  
× (H_valid-d+1) primary branches

plus secondary operators only at three positions.

Do not launch the old 6900-branch schedule in this stage.

# ==================================================  
15. Deliverables

experiments/r16_p14_stage2b/  
preregistration.yaml  
metric_contract.md  
commands.sh  
chunk_executability/  
actor_events/  
perturbation_qualification/  
replay/  
atlas_pilot/  
operator_audit/  
tests/  
reports/  
decision.json

artifacts/stage2b/  
source freeze  
raw actor event records  
action-chunk executability records  
frozen actor-conditioned perturbations  
replay records  
R(k) branch records  
baseline readouts  
paired metrics  
bootstrap intervals  
negative results  
test results  
SHA256SUMS

Final REPORT.md must clearly separate:

- immutable Stage-1b negative result
- Stage-2A preflight evidence
- action-chunk executability
- actor-conditioned perturbation validity
- actor-history replay validity
- conditional original Track-A pilot
- Track-B oracle replanability pilot
- local-repair/operator result
- learned/deployable evidence, which does not exist in Stage-2B
- novelty status, which remains no higher than the frozen review boundary

Do not claim:

- general R16-P14 validation
- deployable replanability
- VLA result
- π0.5 result
- N3/N4
- accepted idea
