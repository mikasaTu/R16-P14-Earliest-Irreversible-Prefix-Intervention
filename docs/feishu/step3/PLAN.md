<title>step3</title>

继续仓库：

[https://github.com/mikasaTu/R16-P14-Earliest-Irreversible-Prefix-Intervention](https://github.com/mikasaTu/R16-P14-Earliest-Irreversible-Prefix-Intervention)

当前已冻结结论：

Stage-1b decision = KILL_CORE_HYPOTHESIS

禁止删除、覆盖、改写或弱化这个结论。

创建新的独立研究阶段：

R16-P14 Stage-2A  
Conditional Prefix Replication and Safe-Replanability Atlas

本轮同时验证两条彼此分离的假设：

Track A:  
原 R16-P14 的条件化 task-family 版本。

Track B:  
Policy-Relative Replanability Window，  
即在物理安全约束内，选择基础策略最容易重新接管并完成任务的 prefix。

本轮不训练正式 learned irreversibility head，  
不运行 π0.5，  
不运行 world model，  
不声明 N3/N4，  
不声明 accepted。

# ==================================================  
0. Repository and evidence boundary

首先记录当前：

- repository HEAD/tree
- Stage-1 frozen commit/tree
- Stage-1b commit/tree
- REPORT.md SHA256
- decision.json SHA256
- selected_config_paired_metrics.csv SHA256
- LIBERO source/config SHA256
- demonstration SHA256
- current Python/MuJoCo/PyTorch environment

从当前 main 创建：

branch:  
agent/stage2a-safe-replanability-atlas

创建：

experiments/r16_p14_stage2a/  
preregistration.yaml  
hypothesis_contract.md  
novelty_boundary/  
task_screen/  
actor/  
perturbations/  
replay/  
atlas/  
diagnostic_probe/  
reports/  
decision.json  
commands.sh

artifacts/stage2a/

不得修改：

- artifacts/formal_pilot
- artifacts/stage1b
- experiments/r16_p14_libero_stage1
- experiments/r16_p14_libero_stage1b

只使用普通 Git、JSONL、CSV、Markdown 和 SHA256。  
禁止 formal activation、WORM、自定义发布器或大型 provenance 框架。

==================================================

1. Phase 0 — Reproduce the current clue exactly  
==================================================

读取：

artifacts/stage1b/expert_chunk_calibration/  
selected_config_paired_metrics.csv

机械验证以下事实，不加入新解释：

- M0 safe success = 5/6
- M1 safe success = 5/6
- M2 safe success = 2/6
- M0 与 M1 失败样本不同
- M2 没有提供 M0/M1 均失败时的唯一安全成功
- M0/M1 per-sample union = 6/6

输出：

reports/current_clue_reproduction.md  
reports/current_clue_reproduction.json

明确写：

This is calibration-only n=6 hypothesis-generating evidence,  
not algorithm performance.

若数字无法逐行复现，停止为：

BLOCKED_BY_SOURCE_MISMATCH

# ==================================================  
2. Phase 1 — Freeze the two hypotheses

Track A hypothesis:

In a preregistered deferred-invalidation task family,  
continuing a still-valid nominal prefix and replanning at the  
last physically safe state reduces rework without degrading safety  
relative to immediate replanning.

Track B hypothesis:

Physical safety and frozen-policy replanability are different.  
The best replanning prefix is the safe state with maximum probability  
of successful replanning, not necessarily the detection state or  
the last physically safe state.

定义：

d:  
perturbation detection prefix

S(k):  
whether execution from d to k remains free of the explicit target cause violation

C_pi(k):  
safe-success probability after replanning from state s_k

k_last_safe:  
max k with S(k)=1

k_best:  
lexicographic argmax over:

1. safe-success probability
2. negative new non-nominal actions
3. negative policy calls

不得使用 test-time privileged perturbation metadata。

同时写一个 bounded novelty proposal：

Policy-Relative Replanability Window

让两名互不读取对方结论的 reviewer 做独立 primary-literature audit，  
重点对比：

- PACE
- DEHP
- Bernoulli-Continuation Policy / Continue or Replan
- Adaptive Chunking via State-Action Critic
- REMAC
- CheckVLA
- When to Trust Imagination
- option termination / interruption

该 novelty audit 不阻塞本轮机制实验，  
但没有双审前禁止赋予 N3。

# ==================================================  
3. Phase 2 — Task roster and structural eligibility

固定初始任务：

1. put_the_cream_cheese_in_the_bowl
2. put_the_bowl_on_the_plate
3. open_the_middle_drawer_of_the_cabinet
4. open_the_top_drawer_and_put_the_bowl_inside
5. push_the_plate_to_the_front_of_the_stove
6. put_the_bowl_on_the_stove

这些任务包括已知正锚点、负锚点和新 task families。  
不得根据 Stage-2A gain 删除负任务。

每个任务必须在看 intervention outcome 之前做结构检查：

- stable phase anchor 是否可定义
- environment-only perturbation 是否可实现
- injection instant 是否无直接 violation
- chunk 中是否还剩至少 8 个 actions
- perturbation 后是否理论上存在 fresh replanning
- replay 是否可通过 fresh-env prefix reconstruction

最多允许替换一个任务。  
只能因为 simulator/geometry/perturbation 不可实现而替换，  
不能因为结果为负而替换。

输出：

task_screen/eligibility.jsonl  
task_screen/task_roster_frozen.json  
task_screen/report.md

# ==================================================  
4. Phase 3 — Perturbation contract

仅使用 environment-only perturbations。

Transport/release tasks:

- shift movable target/receptacle
- inject before release
- never teleport the manipulated object
- injection instant contact/violation <=10%

Articulation/path tasks:

- insert obstacle into a future swept path
- no contact at injection
- nominal contact/violation occurs 2–8 actions later

Push/placement tasks:

- insert obstacle or target-region blocker after chunk decode
- no immediate collision
- nominal suffix becomes stale only later

Calibration grid 的选择只依据：

- immediate violation <=10%
- delayed nominal violation between 30% and 80%
- replay >=99%
- not proposed-method gain

splits:

calibration seeds  
evaluation seeds  
held-out seeds

三者必须互斥。

若一个任务没有 qualifying configuration，  
保留负结果，不在 evaluation split 上重新调参。

# ==================================================  
5. Phase 4 — Strong small actor

保留旧 ChunkedBCMLP 作为 weak reference。

新增且只允许新增一个模型：

History-Conditioned State ACT

Input:

- last 4 state observations
- last 3 executed actions
- task ID

Output:

- 16-step action chunk

Constraints:

- parameters <=10M
- no RGB
- no world model
- no privileged input
- 3 training seeds: 7, 17, 29
- no hyperparameter sweep
- same demonstration budget across tasks

固定一个配置，在训练前写入 preregistration。

Actor qualification per task:

A. clean success between 30% and 90%

or

B. target late-phase reach rate >=70%,  
with failures occurring mainly after the target phase is reached.

必须报告：

- grasp/open phase reach
- lift/transport phase reach
- pre-release/contact phase reach
- final task success
- chunk smoothness
- policy calls

若少于 3 个任务 actor qualified：

overall = BLOCKED_BY_ACTOR

不得用弱 actor 结果否定 Track A 或 Track B。

# ==================================================  
6. Phase 5 — Replay gate

继续使用 Stage-1b 已验证的：

fresh environment  
→ restore demonstration/episode anchor  
→ replay exact nominal prefix  
→ inject perturbation at d  
→ replay to branch point  
→ start independent branch

禁止使用旧 snapshot restore 作为主 initializer。

每个 branch point 重建至少 3 次。

要求：

- branch-point state match >=99%
- contact/outcome agreement =100%
- max state error <=1e-9 where exact state is supported
- no shared mutable env across methods
- action chunk hash exact

失败则：

BLOCKED_BY_REPLAY

# ==================================================  
7. Phase 6 — Build the Safe-Replanability Atlas

对每个 qualified task：

- at least 30 evaluation perturbation events
- at least 20 held-out events
- at least 2 actor seeds
- chunk length 16
- prefix stride 1

在 perturbation detection prefix d 后，  
对所有 k=d,...,15 执行：

R(k):  
continue frozen nominal actions through k,  
then call the same frozen actor from s_k.

对每个 R(k) 使用 3 independent policy sampling/training seeds。

记录：

- pre-replan cause violation
- task success
- safe success
- new non-nominal actions
- discarded nominal actions
- completion steps
- policy calls
- path length
- contacts
- gripper events
- final state
- simulator and observation hashes

构造：

S(k)  
C_pi(k)  
cost(k)  
k_last_safe  
k_best

如果 pre-replan 已发生 cause violation，  
该 k 不允许进入 safe set。

# ==================================================  
8. Methods compared from the same atlas

B0:  
immediate replan at d

B1:  
fixed delay 1 then replan

B2:  
fixed delay 2 then replan

B3:  
fixed delay 4 then replan

B4:  
fixed delay 8 then replan

B5:  
PACE boundary then replan

A_original:  
replan at k_last_safe

N_oracle:  
replan at k_best

Secondary controls only:

- hold + replan
- bounded rollback + replan
- cause-specific local repair

Secondary operators 只在：

d  
k_best  
k_last_safe

三个位置运行，避免 branch explosion。

所有方法使用：

- same actor
- same environment seed
- same perturbation
- same execution budget
- same maximum policy calls
- same branch horizon

# ==================================================  
9. Track-A evaluation

Track A primary comparison:

A_original vs B0

必须报告：

- cause violation
- safe success
- post-detection retained nominal steps
- new non-nominal actions
- policy calls
- path length
- completion steps

Track A PASS requires:

1. at least 2 tasks median k_last_safe-d >=2
2. paired 95% CI for safe-success difference lower bound > -0.03
3. new non-nominal actions reduction >=15%
4. direction consistent on at least 2 actor seeds
5. effect present under at least 2 perturbation severities

若通过，结论只能写：

conditional_task_family_support

不得撤销 Stage-1b universal-hypothesis kill。

# ==================================================  
10. Track-B evaluation

Track B primary comparison:

N_oracle vs strongest of B0–B5 and A_original

Track B oracle-signal PASS requires:

1. safe success improves by >=10 percentage points  
OR cause violation falls by >=25% relatively
2. k_best differs from d in >=20% of events
3. k_best differs from k_last_safe in >=20% of events
4. improvement appears in at least 2 tasks or 2 task families
5. improvement is not explained only by extra policy calls

If no oracle gap exists:

Track B = FAIL_NO_REPLANABILITY_SIGNAL

Do not train any head.

# ==================================================  
11. Cheap diagnostic predictability probe

Only if Track B oracle-signal PASS.

Train only simple diagnostic probes:

- logistic regression
- shallow decision tree
- optional two-layer MLP

Inputs must be deployable:

- proprio
- object-target relative state
- gripper state
- contact state
- task phase
- remaining chunk actions
- action speed/curvature
- prefix age since detection

Targets:

- C_pi(k)
- whether k is in the top safe replanability set

Baselines:

- phase only
- cause only
- speed only
- random
- shuffled state
- shuffled labels
- last-safe rule

Use task-disjoint or leave-one-task-out evaluation where possible.

Probe PASS requires:

1. recover at least 40% of oracle gap over best fixed baseline
2. paired CI lower bound >0
3. shuffled-feature benefit collapses
4. no test-time privileged state

This probe is not a deployable algorithm and must not be called  
a learned R16-P14 method.

# ==================================================  
12. Secondary operator-diversity audit

For full replan, hold, rollback, and local repair,  
compute unique-safe-winner frequency.

Activate a future operator-router direction only if:

- at least 2 operators each uniquely win on >=10% held-out events
- the pattern appears in >=2 tasks
- all operator budgets are matched

Otherwise conclude:

NO_OPERATOR_ROUTING_SIGNAL

Do not continue local-repair development merely because it uses fewer actions.

# ==================================================  
13. Statistics

Use:

- paired episode bootstrap
- task-stratified bootstrap
- 95% confidence intervals
- 10,000 resamples
- frozen bootstrap seed

Report both:

- macro average across tasks
- per-task result

Do not pool thousands of branch rows as independent samples.  
The unit of statistical independence is episode/event, not prefix branch.

# ==================================================  
14. Final decision schema

decision.json must contain separate fields:

stage1b_universal_hypothesis:  
KILLED_IMMUTABLE

track_a_conditional_original:  
PASS | FAIL | INCONCLUSIVE

track_b_replanability:  
PASS | FAIL_NO_ORACLE_GAP | FAIL_UNPREDICTABLE | INCONCLUSIVE

operator_router:  
NOT_ACTIVATED | PASS_SIGNAL | FAIL_NO_DIVERSITY

overall:  
PROCEED_TO_LEARNED_REPLANABILITY  
PROCEED_CONDITIONAL_ORIGINAL_ONLY  
STOP_PREFIX_TIMING_FAMILY  
BLOCKED_BY_ACTOR  
BLOCKED_BY_REPLAY  
BLOCKED_BY_INFRA

Do not use accepted=true in this stage.

# ==================================================  
15. Resource limits

- CPU first for branch replay
- maximum 2 GPUs
- maximum 2 concurrent GPU workers
- no π0.5
- no DINO-WM
- no world model
- no large sweep
- no HTML
- no formal activation infrastructure

# ==================================================  
16. First checkpoint

Stop and report before full Atlas launch after completing:

- Phase 0 clue reproduction
- task structural eligibility
- frozen perturbation grids
- actor training smoke
- clean/phase-reach actor gate
- replay gate on at least one positive and one negative anchor

First checkpoint must include:

1. exact HEAD/tree
2. current clue reproduction
3. frozen task roster
4. actor config and parameter count
5. clean success and phase reach per task
6. replay determinism
7. perturbation qualification counts
8. estimated branch-run volume
9. current blocker
10. exact next command

Do not begin full Atlas before this checkpoint is written.

# ==================================================  
17. Deliverables

experiments/r16_p14_stage2a/  
preregistration.yaml  
hypothesis_contract.md  
novelty_boundary/  
task_screen/  
actor/  
perturbations/  
replay/  
atlas/  
diagnostic_probe/  
reports/  
decision.json  
commands.sh

artifacts/stage2a/  
raw branch JSONL  
actor checkpoints  
task roster  
perturbation schedules  
replay evidence  
atlas curves  
paired metrics  
bootstrap results  
negative results  
SHA256SUMS

Final REPORT.md must clearly separate:

- old Stage-1b immutable negative result
- conditional original-track result
- replanability oracle signal
- diagnostic predictability
- operator-diversity result
- learned/deployable evidence, which does not exist unless a later stage runs
