# step2

继续项目：



https://github.com/mikasaTu/R16-P14-Earliest-Irreversible-Prefix-Intervention



当前 Stage-1 正式结论为：



REVISE_TASKS_OR_PERTURBATIONS



当前结果不能进入 learned prefix-risk model。下一轮命名为：



R16-P14 Stage-1b:

Contract Repair, Expert-Chunk Calibration, and Revised Oracle Audit



本轮不得训练 learned risk head，不得运行 π0.5，不得运行 world model，

不得修改 novelty grade，不得宣称最终算法有效。



先阅读并以当前仓库真实内容为准：



- README.md
- docs/EXPERIMENT_REPORT.md
- experiments/r16_p14_libero_stage1/preregistration.yaml
- r16_p14_stage1/oracle_audit.py
- r16_p14_stage1/aggregate.py
- r16_p14_stage1/envs.py
- r16_p14_stage1/perturbations.py
- current raw oracle records and clean rollout artifacts

从当前 main HEAD 创建独立 branch/worktree。

不得覆盖或修改 Stage-1 原始 artifacts。

创建新目录：



experiments/r16_p14_libero_stage1b/

artifacts/stage1b/



只使用普通 Git、JSONL、CSV、Markdown 和 SHA256。

禁止增加 formal activation、WORM、复杂发布器或无关安全基础设施。



==================================================

1. 本轮科学问题

==================================================



本轮必须独立回答：



Q1. 当前 Stage-1 的正信号在修正指标定义后是否仍成立？



Q2. Bowl replay determinism 低于 99% 的原因能否被修复？



Q3. 在完全不依赖 BC 的 expert action chunk 上，是否存在：

    perturbation detection

    → nonzero safe prefix

    → point of no return

    的真实窗口？



Q4. 相比 perturbation 一出现就 immediate full replan，

    继续执行到最后安全 prefix 是否减少返工？



Q5. 在相同最后安全位置，cause-specific local repair

    是否优于 full replan？



Q6. 若 expert chunk 上机制成立，更强的小型 chunk policy

    是否仍能复现该现象？



==================================================

1. 冻结和证据边界

==================================================



首先冻结：



- current repository HEAD/tree
- original Stage-1 frozen source commit/tree
- current raw records SHA256
- LIBERO version/config
- demonstration files
- existing checkpoints
- task names
- policy seeds
- simulator and Python environment

创建：



experiments/r16_p14_libero_stage1b/

  preregistration.yaml

  metric_contract.md

  commands.sh

  source_manifest.json

  offline_reanalysis/

  replay_gate/

  expert_chunk_calibration/

  policy_baseline/

  revised_oracle/

  tests/

  reports/

  decision.json



最终 decision 只能是：



PROCEED_TO_LEARNED_RISK

REVISE_POLICY_SUBSTRATE

REVISE_TASKS_ONCE

KILL_CORE_HYPOTHESIS

BLOCKED_BY_REPLAY

BLOCKED_BY_INFRA



==================================================

1. Phase A — Offline reanalysis of existing Stage-1

==================================================



本阶段不得启动 simulator 或 GPU。



重新读取现有 90 条 oracle candidate records。



修正以下定义：



A. cause_violation



只统计明确的目标 safety/failure cause。

不得再把 “branch 没有在预算内完成任务” 自动计为 safety violation。



分别记录：



- cause_violation
- task_failure
- timeout
- safe_success

B. detection time



d = perturbation insertion/detection prefix。



C. latest safe prefix



k_last_safe 是 d 之后最后一个仍存在 deployable successful intervention

的位置。



D. post-detection retention



retention_after_detection =

(k_last_safe - d) / (chunk_length - d)



不得继续使用 k_last_safe / chunk_length 作为主 retention 指标。



E. 三组 paired comparison



M0 immediate_full_replan_at_d



在 d 立刻丢弃全部剩余 chunk 并重新规划。



M1 continue_to_last_safe_then_full_replan



继续执行原 chunk 从 d 到 k_last_safe，

然后完整重新规划。



M2 continue_to_last_safe_then_cause_specific_repair



继续执行到同一个 k_last_safe，

然后执行 cause-specific local repair。



定义：



timing_gain = M1 - M0

operator_gain = M2 - M1

total_gain = M2 - M0



比较：



- cause violation
- safe success
- new non-nominal actions after detection
- discarded nominal actions
- policy calls
- path length
- completion steps

“rework”必须定义为 perturbation detection 之后需要新生成/新执行的

非原 nominal-safe-prefix 动作数，不能简单等于 branch 总步数。



输出：



offline_reanalysis/reanalysis.jsonl

offline_reanalysis/summary.json

offline_reanalysis/paired_metrics.csv

offline_reanalysis/report.md



必须明确说明当前 Bowl 正信号在新定义下是否仍存在。



在完成该步骤前不得修改任务或 perturbation。



==================================================

1. Phase B — Replay reconstruction repair

==================================================



当前中间 snapshot restore 不得直接作为唯一 branch initializer。



实现 reconstruction-based branch state：



fresh environment

→ restore demonstration phase anchor

→ replay exact nominal chunk prefix

→ at the registered insertion step apply the exact perturbation

→ continue replay to requested branch prefix

→ verify branch-point state

→ start intervention branch



每个 branch 必须使用 fresh env 或完整独立 env instance，

不得在多个 operator 间复用已被执行过的 mutable env。



对每个 branch point 保存：



- full simulator state/hash
- policy observation feature/hash
- object joint poses
- robot state
- contact pairs/hash
- task phase
- perturbation state
- success flag
- controller-visible observation

首先只在 Bowl 的全部 30 candidates 上测试。

每个 branch point 重建 5 次。



Replay gate：



- branch-point state/hash exact where supported
- contact sequence and outcome 100% match
- final-state max abs error within preregistered tolerance
- overall pass rate >=99%

同时保留旧 snapshot_restore 作为对照，报告：



snapshot_restore

vs

prefix_reconstruction



如果 reconstruction 后 Bowl replay 仍低于 99%，停止并写：



BLOCKED_BY_REPLAY



不得继续做 expert calibration 或训练 policy。



==================================================

1. Phase C — Expert action-chunk feasibility audit

==================================================



只有 replay gate 通过后才执行。



本阶段完全不使用 BC policy。



从 demonstration 中取真实 expert chunk：



A_expert = actions[t:t+16]



在相同 phase anchors 上注入 environment-only perturbation，

然后执行与 Stage-1 相同的 branch audit。



将 demonstration 分成不重叠的：



- perturbation calibration split
- evaluation split
- held-out split

不得在 evaluation split 上调整 timing、severity 或 repair primitive。



任务处理：



A. put_the_bowl_on_the_plate



保留为主任务。

perturbation 必须发生在：



- bowl 已稳定 lift
- 已有至少 2 个安全 transport actions
- release 尚未发生

移动 plate/target，不移动被抓 bowl。



B. open_the_middle_drawer_of_the_cabinet



重新标定 obstacle placement 和 timing。



要求：



- obstacle insertion 时没有 contact
- nominal suffix 在 2–8 actions 后才发生 fixture-obstacle contact
- 至少一个 retract/stop/replan operator 在接触前可恢复

C. put_the_wine_bottle_on_the_rack



禁止继续通过直接 teleport manipulated bottle 制造主 perturbation。



优先选择：



- target/rack shift
- stable-grasp 后的 bounded external disturbance
- 不会在注入瞬间直接 drop 的 contact misalignment

若环境不支持合理 target-side perturbation，则标记该任务不适用，

并从 LIBERO-GOAL 中选择一个具有：



stable early prefix

\+

late target/contact/release failure



的新任务。



不得为了得到正结果无限搜索任务。

最多允许一次 bounded replacement。



Calibration 参数选择规则必须在运行前写入 preregistration：



合格 perturbation 需要：



- injection instant immediate violation rate <=10%
- nominal delayed violation rate between 30% and 80%
- median candidate window target range 2–8 actions
- no privileged perturbation metadata exposed to the tested policy

不要以 proposed method 的最终 gain 最大为标准选择扰动。



完成 calibration 后冻结参数，再在 disjoint evaluation demos 上运行

至少 30 个 candidates/task。



Expert gate：



- replay >=99%
- 至少 2 个任务 median window >=2
- 至少 2 个任务 median post-detection retention >=30%
- oracle cause-violation reduction >=30%
- immediate full replan 与 continue-to-safe 的 timing comparison 非空
- same-trigger full replan 与 local repair 的 operator comparison 非空

若 expert chunks 在 revised tasks 上仍不能产生上述信号，写：



KILL_CORE_HYPOTHESIS



不得通过换 policy 或训练 risk head继续救该假设。



==================================================

1. Phase D — Policy substrate repair

==================================================



只有 expert gate 通过才执行。



保留当前 ChunkedBCMLP 作为 reference。



只允许增加一个固定配置的更强小型模型：



history-conditioned state chunk policy



建议实现：



- last 4 state observations
- optional last 3 executed actions
- small Transformer or ACT-style chunk decoder
- chunk length 16
- model parameters <=10M
- same demonstrations and normalization
- no RGB
- no privileged simulator state
- no world model
- no risk head

禁止大规模超参数搜索。

最多比较：



1. current MLP
2. one fixed history-conditioned model

每个模型 3 seeds。



Policy substrate 合格条件：



对至少 2 个任务满足任一：



A. clean task success between 30% and 90%



或



B. phase reach rate >=70%，且主要失败发生在目标 late phase，

   不是 episode 开始阶段失败。



必须单独报告：



- grasp/lift/open phase reach
- transport/alignment phase reach
- release/contact completion
- final task success

如果 expert gate 通过但所有 policy 都无法稳定到达目标 phase，写：



REVISE_POLICY_SUBSTRATE



不得把弱 policy 结果用于否定 R16-P14。



==================================================

1. Phase E — Revised policy oracle audit

==================================================



只有 policy substrate 合格才执行。



至少：



- 2 tasks
- 2 policy seeds
- 30 oracle candidates per task/seed
- prefix stride = 1
- equal branch budget

方法：



M0 nominal_continue

M1 immediate_full_replan_at_detection

M2 continue_to_last_safe_then_full_replan

M3 continue_to_last_safe_then_generic_trim

M4 continue_to_last_safe_then_cause_specific_repair

M5 privileged physical recovery upper bound



主指标：



- explicit cause-violation rate
- task success
- safe success
- intervention-window distribution
- post-detection safe-prefix retention
- retained nominal steps
- discarded nominal steps
- new actions required after detection
- additional policy calls
- path length
- premature intervention
- late intervention
- replay determinism

必须分别报告：



timing gain:

M2 vs M1



operator gain:

M4 vs M2



total gain:

M4 vs M1



不能只报告 M4 vs nominal。



使用 paired environment/demo/candidate seeds，并给出 paired bootstrap CI。

本轮仍是 oracle audit，不是 learned detector 的性能实验。



==================================================

1. Final gate

==================================================



PROCEED_TO_LEARNED_RISK 需要同时满足：



1. replay determinism >=99%
2. expert chunks 至少 2 个任务 median window >=2
3. policy chunks 至少 2 个任务 median window >=2
4. median post-detection retention >=30%
5. cause-specific violation 相对 nominal 下降 >=30%
6. 相比 immediate full replan，

new non-nominal actions 至少下降 15%

1. timing gain 和 operator gain 至少各在一个 failure family 上出现
2. 至少两个 policy seeds 方向一致
3. 不依赖 test-time privileged state

如果只有 Bowl 成立，另外任务全部失败，状态不能是 PROCEED。

应为 REVISE_TASKS_ONCE 或 KILL_CORE_HYPOTHESIS，

取决于 expert audit 是否存在第二个任务信号。



==================================================

1. 资源和停止规则

==================================================



Phase A/B/C 优先使用 CPU。

Phase D 最多使用 2 张 GPU，并发最多 2。

不得使用 π0.5、DINO-WM 或任何大规模 VLA。



不得自动进入 learned risk training。

即使最终 decision=PROCEED_TO_LEARNED_RISK，也必须先停下汇报。



第一次 checkpoint 必须在 Phase A + Bowl replay repair 后给出：



1. exact HEAD/tree
2. 当前 90-record corrected reanalysis
3. old vs corrected metric differences
4. Bowl positive signal 是否仍存在
5. snapshot restore vs reconstruction replay
6. replay pass rate
7. exact files changed
8. blocker
9. 下一条 exact command

第一次 checkpoint 前不得训练新模型。



==================================================

1. 交付物

==================================================



必须提交：



- preregistration.yaml
- metric_contract.md
- old-to-new metric comparison
- replay repair report
- expert calibration records
- policy baseline report if executed
- revised oracle raw JSONL
- paired statistics
- negative results
- decision.json
- final REPORT.md
- exact commands and hashes

最终汇报必须严格区分：



- metric correction
- replay correctness
- expert mechanism feasibility
- policy substrate adequacy
- oracle mechanism result
- learned/deployable evidence（本轮不存在）

不得把 expert/oracle success 写成最终算法性能。
