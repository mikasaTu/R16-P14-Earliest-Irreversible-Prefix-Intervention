# 实验报告

## R16-P14 Stage-2E / step7 S0 实验报告

> 本报告是零 rollout、零模型、零 GPU/PAI 的 CPU 离线诊断重分析。它不构成正式正面证据。

## 结论先行

- `diagnostic_only=true`，`formal_positive_evidence_allowed=false`，`new_idea_generated=false`。计划/提交的 PAI job 为 `0/0`；S1 未启动。
- A 共同支撑：`PASS`，G0-1=`FULL_SUPPORT_ADVANTAGE`，A1 精确复现=`True`。
- B 头顶空间：`BLOCKED`，G0-2=`BUDGET_SCAN_FIRST`；现有 artifact 没有至少两个预设 configured budget levels，因此不能把描述性差异解释为预算因果证据。
- C 算子阶梯：`INCONCLUSIVE`，G0-3=`EXPAND_OPERATOR_FAMILY_FIRST`；U4 与既有 atlas exact crosscheck=`True`。
- 总体仍保持 `accepted=false`、novelty ceiling=`N2_ORACLE_PROTOCOL_BOUNDARY_ONLY`。本阶段没有 learned/deployable evidence。

## 不可变的上游事实

- Stage-1b universal hypothesis=`KILLED_IMMUTABLE`；Stage-2B 的 chunk executability 为 PASS，但 actor-conditioned perturbation 与 actor-history replay 均 BLOCKED，Track A/B 均 INCONCLUSIVE。
- Stage-2C qualification、replay contract 均 BLOCKED；其 shared mutable runtime 污染和 33 个 invalid events 作为历史限制保留。Stage-2D 的事件构造结论也不可改写。
- 本 S0 不重新解释、不覆盖这些事实；只在既有机器 artifact 上做有边界的离线重算。

## A：共同支撑与复现

A1 使用 Stage-2C evaluation machine rows 复现冻结表，不把其结果用于 calibration 选择。A2–A5 只使用 calibration rows，并同时保留 96-event `all_contaminated` 与排除 invalid 后的 `replay_valid_subset`。

| cohort | strongest fixed | restricted Δ(kLR−fixed) | restricted 95% CI | full Δ(kLR−fixed) | full 95% CI | inflation |
|-|-|-|-|-|-|-|
| all_contaminated | fixed_delay_8 | 0.0787 | [-0.069444444444444, 0.236111111111111] | 0.0764 | [0.006944444444444, 0.145833333333333] | 0.0023 |
| replay_valid_subset | fixed_delay_8 | 0.0833 | [-0.194444444444444, 0.361111111111111] | 0.0635 | [-0.007936507936508, 0.142857142857143] | 0.0198 |

A5 的 `any` 与 `fraction` 均明确来自 raw `prefix_cause_violation`，cause type set 来自 raw `cause_violation_type`；每 event 固定聚合 180 行，未把 prefix 行当独立样本。

## B：头顶空间（仅描述）

B1 在 stage 内对齐，未跨 stage 做 event join，也未把相同毫米字符串当作同一 severity。Stage-2C 分别标记 `shared_runtime_known_invalid` 和 `shared_runtime_no_detected_violation`；Stage-2D 标记 `fresh_process_verified`。B2 candidate cells 全部列出，但 configured grid levels 不足，故 G0-2=`BUDGET_SCAN_FIRST`，S1 首动作=`BUDGET_SCAN_FIRST`。

Stage-2C 的旧 aggregation 路径先在 `experiments/r16_p14_stage2c/r16_p14_stage2c/aggregate.py::build_atlas` 中按 `(event_instance_id,prefix_k)` 分组，再由同一函数对每个 operator 的三条 actor rows 计算 `C_family`，并用 `R_U=any(C_family>=2/3)`；B 本次只审计这种结构能否在受控预算下形成可解释 headroom，不能用 actual usage 伪造 preset grid。

B1 的已发表 Stage-2C baseline 区间是约 10–16%，Stage-2D 冻结 fixed-delay 描述区间是 0.85–3.39%。这两个区间同时随 configured/remaining budget、perturbation severity 以及 runtime purity 变化；两阶段的 event/cohort、几何参数和执行契约也不同，因此不能把任一差值归因给单一机制。S0 唯一量化的是 Stage-2C 从 `all_contaminated` 排除 replay-invalid 事件得到 `replay_valid_subset` 的 invalid-event-exclusion subset sensitivity（见 `axis_audit.json`）；由于两 cohort 的 event 集合不同，这不是 paired event effect，也不是因果分解，budget、severity、runtime 的 causal contribution 均为 `NOT_IDENTIFIABLE`。

## C：算子相对 recoverability 阶梯

- `all_contaminated`：events=96，U1/U2/U3/U4 defined=21/29/38/39，both-defined=21，Type-7 absolute shift quantiles=[0.0, 0.0, 1.0, 4.0, 12.0]，|shift|≥1 fraction=0.5238，monotonic violations=0。
- `replay_valid_subset`：events=63，U1/U2/U3/U4 defined=8/12/12/12，both-defined=8，Type-7 absolute shift quantiles=[0.0, 0.0, 1.5, 4.0, 7.0]，|shift|≥1 fraction=0.6250，monotonic violations=0。

C 的 U4 与既有 atlas 逐 event crosscheck 为 `True`。集合嵌套使单调性在代数上应恒真；本次 violation 被作为实现/缺失/NaN 审计项，而非直接污染证明。valid cohort 的 U1 support 若不足 10，则 G0-3 只能为 `EXPAND_OPERATOR_FAMILY_FIRST`。None 只在单调性比较使用 sentinel d−1=1，不进入安全成功或位移。

## 机制反解（observation → code trace → mechanism → falsifier）

1. observation：Stage-2C 不同 operator 的 C_family / boundary 会变化。code trace（仅 Stage-2C）：`experiments/r16_p14_stage2c/r16_p14_stage2c/aggregate.py::build_atlas`（operator actor-mean 与 `R_U`），以及 `experiments/r16_p14_stage2c/r16_p14_stage2c/runtime.py::reconstruct_to_prefix`、`::rollback_action`、`::hold_action`（prefix 重放与 prelude）。mechanism：差异首先来自 hold/rollback 的 action prelude 与 fresh_h4/fresh_h16 的 horizon/调用预算改变，而不是新的学习器。falsifier：下一阶段必须在两个以上预设 configured budget levels、相同事件和相同 tail 下做受控 factorial 对齐；S0 不实现或运行它。
2. observation：A 的 boundary baseline 有定义事件较少，fallback 后 full-support 数字与 restricted 数字可能不同。code trace（仅 Stage-2C）：`experiments/r16_p14_stage2c/r16_p14_stage2c/aggregate.py::build_atlas` 的 boundary loop（`k_last_recoverable`/`prefix_safety_valid`）和本脚本 `::_event_method_rows`（None→d=2/fresh_h16 fallback）。mechanism：restricted/full 差异可由 support selection 与 shared-runtime contamination 驱动，不能直接当成真实 selector 增益。falsifier：在 fresh-process、共同预算且 outcome-blind 的新测量中，预注册同一 cluster bootstrap 后仍需保持方向。
3. observation：Stage-2D 的 matched/cached/fresh 执行有不同 actual calls。code trace（仅 Stage-2D）：`experiments/r16_p14_stage2d/r16_p14_stage2d/runtime.py::arm_plan`、`::execute_branch` 的 mode 分支与 tail loop；该函数在成功分支不 padding tail calls。mechanism：actual call/time 差异是执行路径结果，不等价于 configured budget 或因果 headroom；B2 因缺受控 grid fail-closed。falsifier：至少两个 preset budgets、相同 tail/action/call cap 的 fresh-process matrix。

## 边界与停止条件

- C/B/A 均为 diagnostic-only 离线结果；不得写成 VLA、训练、闭环或 accepted=true 证据。
- 本阶段不启动 S1，不做 simulator/model/rollout；任何需要这些资源的问题登记为待测。
- `new_idea_generated=false`；没有生成新 idea，也没有将机制说明包装成部署主张。
- 数据输入、hash、官方 Stage-2D sharded loader、selection receipt、测试和 checksum 见相邻 artifacts。

## 飞书归档核对补充（Stage2E S0）

本节仅用于把机器报告中的关键审计量完整归档到本云文档；不改变前文结论。当前分析提交为 `486c1232`；后续 LF seal 只会处理文件行尾/封存一致性，不改变科学结果。

### A1：九项精确复现

以下九项均来自冻结 Stage-2C machine table，分母/分子均按原表复现；`exact_match=true`。这些是 diagnostic-only 复现，不是正式正面证据：

| baseline | numerator / denominator | safe-success |
|-|-|-|
| action_disagreement | 10 / 144 | 6.9% |
| fixed_delay_1 | 8 / 144 | 5.6% |
| fixed_delay_2 | 13 / 144 | 9.0% |
| fixed_delay_4 | 15 / 144 | 10.4% |
| fixed_delay_8 | 16 / 144 | 11.1% |
| immediate_fresh_h16 | 23 / 144 | 16.0% |
| k_last_observed_safe | 3 / 72 | 4.2% |
| k_last_recoverable | 20 / 54 | 37.0% |
| velocity_phase | 16 / 144 | 11.1% |

### A2–A4：共同支撑、全支撑回落与 CI

`k_last_recoverable` 的 restricted 支撑覆盖 21/48 events；full-support 规则对未定义 boundary 明确回落到 `k=d`（immediate/fresh_h16）。聚类 bootstrap 单元为 `(task, init_state_id)`，10,000 次，seed `216214`，先对 event 内 actor rows 求均值，再等权聚类。

| cohort | restricted Δ(kLR−fixed_delay_8) | restricted 95% CI | full-support Δ(kLR−fixed_delay_8) | full-support 95% CI | support |
|-|-|-|-|-|-|
| all_contaminated（96 events 中排除 invalid=0，仍含 shared-runtime 污染） | 0.0787 | [-0.0694, 0.2361] | 0.0764 | [0.0069, 0.1458] | 21/48 → 48/48 |
| replay_valid_subset（排除 33 个 invalid，63 events） | 0.0833 | [-0.1944, 0.3611] | 0.0635 | [-0.0079, 0.1429] | 9/34 → 34/34 |

历史 Track B held-out safe-success difference `−0.0944` 与上表并列保留；不能将二者当作同一 cohort 的 paired 估计。全支撑下的优势会因 fallback/support 定义而变化，且输入本身包含 shared-runtime 污染，因此不能转写成可部署 selector 增益。

A5 的覆盖分解明确记录为：all contaminated 中 covered `21`、uncovered `27`；replay-valid 子集中的 covered `9`、uncovered `25`。覆盖与未覆盖 event 的 cause type、prefix cause、parameter/task 分布均从 raw rows 逐 event 聚合；没有把 prefix 行当独立样本，也没有读取 Stage-2D reserve/evaluation outcome。

### B：头顶空间判决

B1 仅作 stage 内描述性对齐，未执行跨阶段 event join；Stage-2C 标记 `shared_runtime_known_invalid` / `shared_runtime_no_detected_violation`，Stage-2D 标记 `fresh_process_verified`。Stage-2C 的 event-remaining-horizon / padded event budget 不是预注册的多水平 controlled grid，且 tail/operator/severity/runtime 均不相同。因此预算、severity、runtime 的 causal contribution 均为 `NOT_IDENTIFIABLE`。

B2/B3 的独立判决为 **`BUDGET_SCAN_FIRST`**：冻结 artifact 内不存在足够的预设 budget levels，不能给出一个有因果解释的固定 action/call/tail 数值；S1 第一项应先做预算扫描，而不是直接运行 atlas。该结果不是“没有任何描述性差异”，而是没有可解释的受控预算工作区间。

### C：算子支撑与位移

嵌套阶梯为 `U1={fresh_h4}`、`U2=U1∪{fresh_h16}`、`U3=U2∪{hold_one_step_then_fresh_h16}`、`U4=U3∪{rollback_one_step_then_fresh_h16}`。all contaminated 的 defined counts 为 `21/29/38/39`，both-defined=`21`，`|U4−U1|≥1` 比例 `0.5238`，位移五数摘要 `[0,0,1,4,12]`，monotonicity violations=`0`。replay-valid subset 为 `8/12/12/12`，both-defined=`8`，比例 `0.6250`，摘要 `[0,0,1.5,4,7]`，violations=`0`；U1 valid support 只有 8，不足预注册的 10-event 解释门槛。因此 G0-3=`EXPAND_OPERATOR_FAMILY_FIRST`，而不是声称算子相对性已被正式验证。

### 不可变结论

Stage-1b universal hypothesis=`KILLED_IMMUTABLE`；Stage-2B 的 chunk executability=`PASS`、actor-conditioned perturbation=`BLOCKED`、actor-history replay=`BLOCKED`、Track A/B=`INCONCLUSIVE`；Stage-2C qualification/replay contract=`BLOCKED`；Stage-2D event construction=`BLOCKED`，其 fresh-process branch isolation PASS（54 rows，最大重建误差 0.0）不被本步覆盖。`accepted=false`，novelty ceiling=`N2_ORACLE_PROTOCOL_BOUNDARY_ONLY`，本步不产生 learned/deployable evidence。

### 机制反解（仅解释已有观测，不生成新 idea）

- **boundary 增益的选择效应**：`stage2c/.../aggregate.py::build_atlas` 先按 event/prefix/operator 汇总并只在 boundary 有定义的 event 上形成 `k_last_recoverable`；本步的 `::_event_method_rows` 对 `None` 使用 `d=2` 的 immediate fallback。于是 restricted 与 full 的差异可由 support selection 与 shared-runtime contamination 共同造成；不能称为真实 selector 增益。反证/消歧需要 fresh-process、同一事件、同一预算的预注册 factorial 重测。
- **operator 位移的执行路径来源**：`stage2c/.../runtime.py::reconstruct_to_prefix`、`hold_action`、`rollback_action` 与 `aggregate.py` 的 actor-mean/`R_U=any(C_family>=2/3)` 组合，使 `fresh_h4/fresh_h16/hold/rollback` 改变 action prelude、horizon 与 policy-call 路径；因此边界变化首先说明 operator reach/预算定义不同，不自动说明环境中的一般机制。下一步须用至少两个 configured budget levels 和共同 tail 做受控对齐。
- **污染与可重复性的区别**：Stage-2C 的 shared mutable runtime 会让 `S_obs`/boundary 的一致性受状态复用影响；Stage-2D 的 fresh-process 路径在 54 rows 上通过隔离，但不能把那一结果外推为 Stage-2C 原始行已被净化。S1 需记录完整接触拓扑、进程/env/chunk hashes，并以纯函数离线重算 cause。

每个机制的可证伪条件都是：在新阶段先冻结预算、tail、operator、split 和 cluster bootstrap，再观察方向是否在完整支撑、clean replay 与两任务中复现；本 S0 不运行 rollout，也不把任何方向升级为新 idea。

### 代码、产物与资源边界

- 复现脚本：`/workspace/leon/.../scripts/verify_r16p14_stage2e_s0.py`（CPU 可运行）。
- Stage2E 代码与报告：`experiments/r16_p14_stage2e/`。
- 机器产物：`artifacts/stage2e/s0/common_support/`、`headroom/`、`operator_ladder/`，校验清单为 `artifacts/stage2e/SHA256SUMS`。
- 只读输入：Stage-2C/Stage-2D 冻结 artifacts；没有改写任何旧阶段目录。
- 计划/提交 PAI jobs=`0/0`；没有 GPU、模型加载、rollout 或训练，故不涉及 09:30–09:40、19:30–19:40 的 PAI 停止/恢复窗口。

**最终边界：** S0 已完成 CPU 离线重分析与规范输出，S1 未启动；不应宣称正面正式证据、accepted、VLA 验证或可部署方法。

