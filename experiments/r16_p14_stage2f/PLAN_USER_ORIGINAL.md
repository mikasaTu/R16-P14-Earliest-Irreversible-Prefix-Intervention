# 任务：R16-P14 Stage-2F / step8 — S1 自然失败事件源与非嵌套算子族交叉检验

## 背景（只读事实，不要重新推导，不要质疑）
- 仓库 mikasaTu/R16-P14-Earliest-Irreversible-Prefix-Intervention，main 分支
- 冻结不可改写：Stage-1b=KILL_CORE_HYPOTHESIS（universal hypothesis KILLED_IMMUTABLE）；
  Stage-2B=BLOCKED_BY_ACTOR_CONDITIONED_PERTURBATION；Stage-2C=BLOCKED_BY_REPLAY_CONTRACT；
  Stage-2D=BLOCKED_BY_EVENT_CONSTRUCTION；Stage-2E-S0 全部产物 diagnostic_only
- Stage-2D 的 fresh-process 分支隔离已 PASS（spawn 独立进程、fresh env、
  重建四状态三动作历史、最大重建误差 0.0）。直接 import
  experiments/r16_p14_stage2d/r16_p14_stage2d/fresh_process.py 与 isolation.py，
  不重写、不"改进"
- 恢复算子实现已存在于 experiments/r16_p14_stage2c/r16_p14_stage2c/runtime.py
  （reconstruct_to_prefix / hold_action / rollback_action）。注意
  reconstruct_to_prefix 目前带 perturbation parameter 参数，本阶段必须走
  零注入路径；若无法在不改动 2C 冻结代码的前提下做到，就在 stage2f 下新写
  一个零注入版本，不要改 2C
- ACT actor 是 shared multitask，checkpoint 在 artifacts/stage2a/actor/checkpoints/
  seed_{7,17,29}.pt，覆盖全部六个 LIBERO-GOAL 任务
- 本阶段不注入任何扰动，不训练 actor，不引入 VLA/世界模型/RGB

## 本阶段的主张（写清楚，不要漂移）
主张不是"保留安全前缀能赢 baseline"。那条在 Stage-2E-S0 已被证据削弱到
无可测优势，本阶段不再测它，但 immediate_fresh 必须作为参考臂全程记录。
本阶段唯一要测的主张是：可恢复边界 k* 是相对恢复算子族 U 定义的量，
且这种相对性是非平凡的——不是同一个隐变量被不同阈值切了一刀。

## 硬约束
1. PAI 作业上限 2 个，单作业 ≤2×A800。总 GPU 预算上限 20 小时，超出即停并报告
2. artifacts/stage2{a,b,c,d,e}/ 与 experiments/r16_p14_stage2{a,b,c,d,e}/ 一律
   只读冻结。新产物只写 experiments/r16_p14_stage2f/ 与 artifacts/stage2f/
3. 零注入。任何 perturbation / target shift / path blocker 代码路径都不得被调用。
   若发现某个复用函数隐式注入，停下报告，不要绕过
4. 每个 branch 必须是 multiprocessing spawn 的独立进程，独立 env，
   不共享 wrapper/RNG/cache。缺 pid / env_hash / chunk_hash 任一，该 branch BLOCKED
5. 严格执行 experiments/r16_p14_stage2e/INSTRUMENTATION_S1.md 的 D1–D5。
   特别是 D1：每步记录规范化排序后的完整 contact geom-name pair 集合，
   contact_count 不能替代
6. calibration / evaluation 严格分离。所有阈值、预算档、算子族、判据的选择
   只能在 calibration 上做；evaluation outcome 在 Phase 2 的 selection receipt
   写完并提交之前为 open-deny
7. K1/K2/K3 的数值判据必须在看任何 Phase 数据之前写进
   experiments/r16_p14_stage2f/PREREG_S1.md 并 commit，之后不得修改
8. 任一 K 门失败即停，不得继续下游 Phase，不得"诊断性继续执行"。
   这一条覆盖 Stage-2D 那次外部指令导致的全矩阵强跑
9. 每个产出文件记 SHA256 到 artifacts/stage2f/SHA256SUMS
10. 前置条件不满足就停下写报告，不要自己补数据、插值或重新划分

## Phase 0A — CPU 前置修正（零 rollout，先做）
Stage-2E-S0 的 A 把对照集限定成 {fixed_delay_1,2,4,8}，漏掉了
immediate_fresh_h16（16.7%，强于 fixed_delay_8 的 11.8%），而 immediate_fresh
正是 idea 原文里"同触发时刻全量重规划"那条对照。
0A.1 复用 scripts/run_r16p14_stage2e_s0.py 的 A 路径，把对照集扩成
     {fixed_delay_1,2,4,8, immediate_fresh_h16}，重跑 G0-1，输出到
     artifacts/stage2f/phase0a/
0A.2 同时报告绝对 safe success 水平（受限支撑与全支撑各一列），
     不要只报 Δ。S0 的报告只报了 Δ 和 inflation，隐去了 38.1%→19.4% 的绝对塌陷
0A.3 把结论一句话写进 PREREG_S1.md 的背景节。若对 immediate_fresh 无优势，
     PREREG 必须明确写明本阶段不再主张性能赢面

## Phase 0B — 自然失败事件采集（GPU）
任务固定为 put_the_cream_cheese_in_the_bowl 与 put_the_bowl_on_the_plate。
put_the_bowl_on_the_stove 明确排除：artifacts/stage2a/actor/eval 的 30 个
clean episode 里 lift 之后失败为 0。
0B.1 冻结 init pool：每任务 100 个 reset-randomized state，记 hash，
     ID 0–9 infrastructure / 10–49 calibration / 50–89 evaluation / 90–99 reserve
0B.2 2 任务 × 3 seed × 100 init 闭环 rollout，零注入
0B.3 事件锚点沿用 Stage-2D 结构判据：grasp + stable_lift_count>=2 +
     gripper closed + task 未成功 + 剩余 chunk 长度 == H_VALID。
     一个 rollout 最多产一个事件
0B.4 每步按 D1 记录完整接触拓扑；cause 标签由离线纯函数产出，至少并列记录
     两个：既有 release-based label，以及 release 前可触发的候选量
     （manipulated object 与非目标 geom 的接触拓扑越界）。运行时不得
     把任一个写死为唯一判据

### K1（硬门）
两个任务各自的 calibration 合格事件数 >= 25 且 evaluation >= 25。
任一不满足 → 状态 BLOCKED_BY_NATURAL_EVENT_YIELD，停，写报告，不进 Phase 1。

## Phase 1 — 预设预算扫描（GPU）
Stage-2E-S0 的 B 门判定冻结数据里没有预设预算网格，只有实际用量，
因此本 Phase 是 S1 的第一个 GPU 动作，不能跳过。
1.1 预设网格：tail_horizon ∈ {4, 8, 16} × action_budget ∈ {8, 16, 32}，
    policy call cap 固定为一个常数并记录。这是 configured grid，
    不得用 event remaining horizon 或实际用量替代
1.2 每任务 20 个 calibration 事件 × 3 recovery actor seed × 4 个恢复算子
    （fresh_h4, fresh_h16, hold_1+fresh_h4, rollback_1+fresh_h16）
    × 前缀点 k ∈ {2,4,8,12,16}
1.3 每格报告 oracle-best safe success、最弱臂、臂间差、cluster 数

### K2（硬门）
存在至少一个预算档，在两个任务上同时满足
oracle_best ∈ [0.25, 0.85] 且 (oracle_best − weakest_arm) >= 0.15。
不存在 → 状态 BLOCKED_BY_NO_WORKING_REGION，停，写报告，不进 Phase 2。
存在多个 → 按预注册顺序取第一个：oracle_best 降序、臂间差降序、
action_budget 升序、tail_horizon 升序。选择写进 selection receipt 并 commit，
之后不得更改。

## Phase 2 — 非嵌套算子族交叉检验（GPU，主实验）
2.1 两个等势非嵌套族，固定不得更改：
      U_A = {fresh_h4,  hold_1 + fresh_h4}
      U_B = {fresh_h16, rollback_1 + fresh_h16}
    U_full = U_A ∪ U_B 作为参考，等于 Stage-2C 的旧 U
2.2 在 K2 选定的单一预算档下，两任务全部 calibration + evaluation 事件
    × 3 recovery actor seed × 4 算子 × 前缀点 k ∈ {2,4,6,8,10,12,14,16}
2.3 参考臂全程记录但不参与任何选择：immediate_fresh、fixed_delay_2/4/8
2.4 对每个事件计算 k*(U_A)、k*(U_B)、k*(U_full)，沿用 Stage-2C 的
    R_U 定义与 max{k : R_U(k)} 语义；无定义记 None，不得用 fallback 伪造
2.5 主统计量：
      crossing_A = 1[k*(U_A) > k*(U_B)]，crossing_B = 1[k*(U_B) > k*(U_A)]
      minority_crossing_rate = min(mean(crossing_A), mean(crossing_B))
      spearman = Spearman(k*(U_A), k*(U_B))，只在两族都有定义的事件上算
2.6 噪声地板（必须做）：对同一个族，把 3 个 recovery actor seed 做 split-half，
    用同样的方式算族内交叉率，重复所有可能的划分，给出零分布。
    跨族 minority_crossing_rate 必须显著高于这个零分布的 95 分位
2.7 报告两族各自的平均 safe success，确认交叉不是围绕一个平局的噪声
2.8 全部统计用 (task, init_state_id) cluster bootstrap，10000 次，seed 216214，
    事件内先对 3 个 actor row 取平均；prefix row 永远不是独立样本

### K3（硬门，本阶段的判决）
在两个任务上同时满足全部三条：
  a) 两族都有定义的事件数 >= 30
  b) minority_crossing_rate >= 0.15，且其 95% CI 下界高于族内 split-half
     零分布的 95 分位
  c) spearman <= 0.8
全部满足 → NONTRIVIAL_OPERATOR_RELATIVITY，S2 可以启动。
任一不满足 → MONOTONE_RESCALING_ONLY，算子相对性是平凡的，
本 idea 的概念主张判负，停，写报告，不要自行设计补救实验。

## 产出
- experiments/r16_p14_stage2f/PREREG_S1.md（Phase 0B 之前 commit，之后不改）
- artifacts/stage2f/phase0a/{summary.json,table.csv}
- artifacts/stage2f/phase0b/{init_pool,events.jsonl,contact_topology/,summary.json}
- artifacts/stage2f/phase1/{grid_rows.jsonl,summary.json,selection_receipt.json}
- artifacts/stage2f/phase2/{atlas_rows.jsonl,boundaries.jsonl,crossing.json,
  null_distribution.json,summary.json}
- experiments/r16_p14_stage2f/reports/REPORT_S1.md（中文）。必须包含：
  每个 K 门的数值与判决、绝对水平而不只是 Δ、两族各自的 safe success、
  噪声地板、以及一节"本阶段没有测的东西"
- experiments/r16_p14_stage2f/pai/jobs.json（run_id、job_id、tree hash、
  GPU 小时、最终状态，失败原样保留不改写）
- artifacts/stage2f/SHA256SUMS
- scripts/verify_r16p14_stage2f.py，在完整克隆上可跑通

## 停止条件
- 任一 K 门失败 → 立即停，不得继续下游 Phase，不得诊断性强跑全矩阵
- GPU 累计超 20 小时 → 停并报告已完成部分
- 发现零注入契约被违反 → 停并报告，不要绕过
- 发现需要 VLA、训练或新基准才能回答的问题 → 登记为 S2 待测，不要自己启动
- 全部完成后不要自行启动 S2