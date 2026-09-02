
# step7

> 执行元数据：Stage2E S0；prereg 首提交 0ff6aef8；pre-outcome 消歧 96f55a18；diagnostic only；零 PAI/rollout；S1 未启动。

# 任务：R16-P14 Stage-2E / step7 — S0 离线共同支撑重分析与算子阶梯可行性（零 rollout）

## 背景（只读事实，不要重新推导，不要质疑）

- 仓库 mikasaTu/R16-P14-Earliest-Irreversible-Prefix-Intervention，main 分支
- 冻结且不可改写的结论，禁止删除、覆盖、弱化或重新解释其中任何一条：  
Stage-1b = KILL_CORE_HYPOTHESIS（universal hypothesis KILLED_IMMUTABLE）  
Stage-2B = BLOCKED_BY_ACTOR_CONDITIONED_PERTURBATION  
Stage-2C = BLOCKED_BY_REPLAY_CONTRACT  
Stage-2D = BLOCKED_BY_EVENT_CONSTRUCTION
- Stage-2D 的 fresh-process 分支隔离已是 PASS（54 rows，最大重建误差 0.0）。  
本步不重写、不评估、不"改进"这套 runtime
- Stage-2C 的 recovery 行受 shared mutable runtime 污染：96 个 event 中 33 个  
replay-invalid；1440 个 recovery cell 中 363 个 S_obs 自相矛盾
- 本步不跑任何 rollout、不训练、不改 runtime、不碰 evaluation outcome

## 本步要回答三个互相独立的问题，各自单独出结论

Q1 共同支撑：k_last_recoverable 的 37.0% 有多少是选择效应  
Q2 头顶空间：在哪个 recovery budget 区间各臂才有可分辨的差距  
Q3 算子相对性：换算子族，k_last_recoverable 到底动不动

## 硬约束

1. 禁止发起任何 GPU 作业、PAI 作业、仿真 rollout、模型加载。全部在 CPU 完成
2. artifacts/stage2c/、artifacts/stage2d/、experiments/r16_p14_stage2{a,b,c,d}/ 与  
docs/feishu/ 一律只读冻结。新产物只写到 experiments/r16_p14_stage2e/ 和  
artifacts/stage2e/
3. 所有输出标 diagnostic_only=true、formal_positive_evidence_allowed=false。  
本步不产生任何正面证据，只产生"下一步该测什么"的判决
4. 2C 的行是被污染的。任何基于 2C 的数字必须同时报告全集与  
replay-valid 子集（63/96 events）两个版本，并显式标注污染状态。不得只报有利的那个
5. 不得读取、聚合、或以任何方式让 evaluation split 的 outcome 参与选择。  
Stage-2D 的 RESERVE_IDS 80–99 保持未读
6. 判据阈值必须在看数据之前写进 experiments/r16_p14_stage2e/PREREG_S0.md 并提交，  
之后不得修改。事后调阈值直接判本步作废
7. 每个产出文件记录 SHA256，追加到 artifacts/stage2e/SHA256SUMS
8. 任何前置条件不满足（字段缺失、行数对不上、split 定义不足）就停下写报告，  
不要自己补数据、不要插值、不要重新划分

## 任务 A — 共同支撑重分析（回答 Q1）

输入：  
artifacts/stage2c/formal_matrix/recovery_operator_rows.jsonl（17280 行）  
artifacts/stage2c/recoverability_atlas/rows.jsonl、event_boundaries.jsonl、invalid_events.jsonl  
artifacts/stage2c/crossfit_replanability/baseline_rows.jsonl、crossfit_rows.jsonl

A1. 复现 2C 报告里那张"可部署 baseline"表（immediate_fresh_h16 16.0%、  
fixed_delay_8 11.1%、k_last_recoverable 37.0% 等）。数字对不上就停下报告差异，  
不要改代码去凑  
A2. 给每个 baseline 加一条显式的全支撑规则：boundary 未定义（k_last_recoverable  
返回 None）时回落到 k=d（immediate）。重算全部 baseline，使每一个都定义在  
全部 48 个 calibration event 上  
A3. 输出三列并排的对照表：受限支撑 / 全支撑 / 支撑覆盖率。对  
k_last_recoverable 相对最强固定延迟 baseline 的 safe-success 差，用  
(task, init_state_id) cluster bootstrap 10000 次、seed 216214 给 95% CI  
A4. 把 Track B 的 held-out −0.0944 与 A3 的 calibration 侧数字并排放。  
显式计算并报告：受限支撑下的优势有多少可以被"只保留存在可恢复 k 的 event"  
这一条解释。方法自选，但必须写清楚  
A5. 分解 21/48 覆盖：未覆盖的 27 个 event 在 cause_violation_type、  
prefix_cause_violation、parameter_id、task 上的分布，与被覆盖的 21 个对比

## 任务 B — 头顶空间审计（回答 Q2）

输入：A 的输入，加上  
artifacts/stage2d/frozen_rule/baselines.json  
artifacts/stage2d/calibration_atlas/rows.jsonl、summary.json  
artifacts/stage2d/confirmatory_evaluation/method_rows.jsonl、paired_rows.jsonl

B1. 把 2C 与 2D 的 safe success 按四个轴对齐成一张表：recovery operator /  
tail horizon 与 action budget / 扰动 severity / runtime 是否干净。  
重点解释 2C 的 10–16% 与 2D 的 0.85–3.39% 之间的差距分别有多少来自  
预算差、severity 差、runtime 污染  
B2. 在这张表上定位一个候选工作区间：oracle-best safe success 落在 [0.25, 0.85]、  
且最弱臂比 oracle-best 低至少 0.15。列出所有满足的 (budget, tail horizon,  
severity) 组合；一个都没有就明确写"冻结数据内不存在可用工作区间"  
B3. 输出给 S1 的预算建议，含 action budget、tail horizon、policy call budget 的  
具体取值和依据。不要给区间，给具体数

## 任务 C — 算子阶梯可行性（回答 Q3）

C1. 从 2C 的 atlas 行构造嵌套算子族阶梯：  
U1 = {fresh_h4}  
U2 = {fresh_h4, fresh_h16}  
U3 = U2 ∪ {hold_one_step_then_fresh_h16}  
U4 = U3 ∪ {rollback_one_step_then_fresh_h16}（= 现有 U）  
对每个 event 重算 k_last_recoverable(Ui) 与 k_irrev_U(Ui)  
C2. 验证单调性：Ui ⊂ Uj 应有 k_last_recoverable(Ui) ≤ k_last_recoverable(Uj)。  
逐 event 检查，报告违反数。违反不为 0 说明 R_U 里有非确定性或污染残留，  
必须定位到具体 event 并报告，不要平均掉  
C3. 报告边界位移分布：|k_last_recoverable(U4) − k_last_recoverable(U1)| 的  
中位数、分位数、以及位移 ≥1 的 event 比例。全集与 replay-valid 子集各一份  
C4. 报告支撑变化：从 U1 到 U4，boundary 有定义的 event 数怎么变。  
如果 U1 下只有个位数 event 有定义，直接写明这条阶梯在统计上不成立  
C5. 基于 C3/C4 写一份 S2 的算子族设计建议：现有四个算子的 reach 差异是否  
足以支撑"换算子族边界就变"这条主张；不够的话需要补哪一类算子  
（给算子的定义，不要给实现）

## 任务 D — S1 仪器化契约（无需数据，写规范）

2C 的行只记了 contact_count 这个标量，没记接触对集合，所以任何新的 cause 定义  
都无法离线重算，只能重跑。为避免 S1 之后重蹈覆辙，写出  
experiments/r16_p14_stage2e/INSTRUMENTATION_S1.md，至少规定：  
D1. 每步必须记录完整接触对集合（geom 名对），不是计数  
D2. cause 判据与行记录解耦：行里存原始接触拓扑与几何量，cause 标签在离线层  
由一个可替换的纯函数计算  
D3. 记录一个能在 release 之前触发的候选量（例如被操作物与非目标 geom 的  
接触拓扑越界），与现有 release-based cause 并存记录，不替换  
D4. 记录 branch 的进程 ID、env 构造 hash、chunk 字节 hash，沿用 Stage-2D 的  
isolation 签名格式

## 产出

- experiments/r16_p14_stage2e/PREREG_S0.md（先于分析提交）
- experiments/r16_p14_stage2e/INSTRUMENTATION_S1.md
- artifacts/stage2e/s0/common_support/{table.csv,summary.json}
- artifacts/stage2e/s0/headroom/{table.csv,summary.json}
- artifacts/stage2e/s0/operator_ladder/{per_event.jsonl,summary.json}
- experiments/r16_p14_stage2e/reports/REPORT_S0.md（中文）
- artifacts/stage2e/SHA256SUMS
- 复现脚本 scripts/verify_r16p14_stage2e_s0.py，CPU 可跑通

## G0 判决（三路独立，各自给结论，不要合并成一个总分）

G0-1 由 A 决定 S1 测哪个主张，不决定 S1 跑不跑：  
(a) 全支撑下优势仍在且 CI 下界 > −0.05 → S1 把 k_last_recoverable 当可部署  
selector 测  
(b) 全支撑下优势归零 → S1 放弃"打赢 baseline"这条，只测描述性的算子相对性  
(c) 全支撑下优势反号 → 记为负结果，"保留安全前缀有增益"这个框架就地作废，  
写清楚并停下等人工决策  
G0-2 由 B 决定：B2 找到工作区间 → S1 直接用 B3 的预算跑；一个都没找到 →  
S1 的第一件事是预算扫描，不是 atlas  
G0-3 由 C 决定：位移 ≥1 的 event 比例 ≥ 30% 且单调性违反 = 0 → 现有算子阶梯  
可用；否则 S2 必须先按 C5 扩宽算子族

## 停止条件

- 出现任何需要 GPU 或 rollout 才能回答的问题 → 停下，写进报告的"S1 待测"一节，  
不要自己跑
- 发现冻结 artifact 与报告数字不一致 → 停下报告，不要修改 artifact，也不要  
修改代码去对齐
- 任何一个任务的前置字段缺失 → 该任务标 BLOCKED 并说明缺什么，其余任务继续
- 全部完成后不要自行开启 S1

