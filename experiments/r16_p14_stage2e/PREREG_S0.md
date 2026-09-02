# R16-P14 Stage-2E / step7：S0 离线共同支撑重分析与算子阶梯可行性

**文档状态：已冻结的 S0 preregistration（首提交）**
**分析模式：零 rollout、零模型、零 GPU/PAI**
**本文件用途：在读取任何本阶段 outcome 之前冻结输入契约、阈值、算法和判决规则**

## 0. 冻结元数据与不可变边界

本文件在任何 Stage-2E 结果读取、解析、聚合或选择之前提交。提交之后，下面的阈值、分母、分组、排序、bootstrap、缺失处理和 G0 规则均不可事后修改。若发现输入或既有报告不一致，只能按本文件记录差异并将相应任务标记为 `BLOCKED`，不得修改原始 artifact、补数据、插值、重新划分或调整阈值。

| 字段 | 冻结值 |
| --- | --- |
| stage | `R16-P14 Stage-2E / step7 / S0` |
| created_utc | `2026-09-02T10:19:46+00:00` |
| repository | `git@github.com:mikasaTu/R16-P14-Earliest-Irreversible-Prefix-Intervention.git` |
| branch | `agent/stage2e-s0-offline-common-support-operator-ladder` |
| worktree | `/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/R16-P14-stage2e-s0-offline-common-support-operator-ladder` |
| immutable parent commit | `edebdfc64576129d994535dacb76de930f493c8d` |
| immutable parent tree | `fa0a951abc1aa778cfa76663679173557e6c9b96` |
| parent verification | `git fetch --prune origin main` over SSH, followed by exact commit/tree readback |
| diagnostic_only | `true` |
| formal_positive_evidence_allowed | `false` |
| rollout_allowed | `false` |
| model_loading_allowed | `false` |
| GPU_allowed | `false` |
| PAI planned/submitted jobs | `0 / 0` |
| evaluation-selection_allowed | `false` |
| Stage-2D `RESERVE_IDS` | `80–99`，保持未读 |
| new_idea_generated | `false` |
| S1 auto-start | `false` |

本提交只允许新增本文件。Stage-2E 其他规范、脚本、artifact、报告和 checksum 必须在后续独立提交中产生；本提交不包含它们，也不在本提交中读取或生成任何 outcome。

### 0.1 既有不可改写事实

以下是只读的上游结论，本阶段不重写、不削弱、不翻转，也不将它们合并成新的总分：

- Stage-1b：`KILL_CORE_HYPOTHESIS`；universal hypothesis = `KILLED_IMMUTABLE`。
- Stage-2B：`BLOCKED_BY_ACTOR_CONDITIONED_PERTURBATION`。
- Stage-2C：`BLOCKED_BY_REPLAY_CONTRACT`。
- Stage-2D：`BLOCKED_BY_EVENT_CONSTRUCTION`。
- Stage-2D 的 fresh-process branch isolation 已有只读结论：54 rows、最大重建误差 `0.0`、隔离通过。
- Stage-2C recovery 行存在 shared mutable runtime 污染：96 个 event 中 33 个 replay-invalid；1440 个 recovery cell 中 363 个 `S_obs` 自相矛盾。

Stage-2E 只回答三个互相独立的离线问题：

1. `k_last_recoverable` 的共同支撑/选择效应（Q1）。
2. 在 recovery budget、tail horizon 和 policy-call budget 上是否存在可分辨的头顶空间（Q2）。
3. 更换嵌套 recovery operator family 时，边界是否移动且是否足够稳定（Q3）。

本阶段不验证任何最终部署策略，不把算子相对边界解释为物理属性，不产生 VLA、训练或闭环证据。

## 1. 读写、数据和过程禁令

### 1.1 只读目录

以下路径在 Stage-2E 全程只读，禁止覆盖、删除、重写、重新打包或建立会改变其内容的兼容副本：

```text
artifacts/stage2c/
artifacts/stage2d/
experiments/r16_p14_stage2a/
experiments/r16_p14_stage2b/
experiments/r16_p14_stage2c/
experiments/r16_p14_stage2d/
docs/feishu/
```

新产物只能写入 `experiments/r16_p14_stage2e/` 与 `artifacts/stage2e/`。本 S0 首提交例外地只写入本文件。

### 1.2 零 rollout 约束

- 不创建环境，不加载 simulator、MuJoCo、robosuite、LIBERO、ACT 或任何 world model。
- 不执行仿真 rollout、物理重放、训练、推理、GPU 程序或 PAI/DLC job。
- 只允许 CPU 上对已冻结 JSONL/CSV/JSON/Markdown 的离线校验和重算。
- 若某一问题必须依赖 GPU、模型或 rollout 才能回答，停止该问题，并在后续中文报告的“待测”部分登记；不得用代理数值替代。
- 本阶段不改 Stage-2D runtime，不实现或训练 selector、risk head、router 或 replanability head。

### 1.3 选择与污染隔离

选择进程必须硬拒绝以下输入：

- Stage-2D evaluation outcome；
- Stage-2D `RESERVE_IDS=80–99` 的任何记录；
- 任何标记为 evaluation、reserve、held-out outcome 的路径；
- 任何将 evaluation 结果作为候选筛选、tie-break、预算选择或阈值选择的调用。

A、B、C 的结果必须按任务独立给出。所有 G0 路径独立运行、独立判决，禁止合成一个总分或以某一路的结果覆盖另一路。

## 2. 全局前置校验契约

所有实际分析脚本必须在读取每一个输入文件前完成下列校验，并把校验结果写入对应任务报告；本文件提交时不读取这些 outcome 文件：

1. **存在性**：所需路径存在、类型正确，且没有用目录 glob 代替冻结 manifest。
2. **SHA256**：对照冻结 source manifest 的完整文件 SHA256；缺 hash、hash 不符或 manifest 不可验证时，该任务 `BLOCKED`。
3. **容器/格式**：每行是合法 JSON object，字段类型与 schema 完全匹配；CSV 列名、JSON key、版本和排序契约严格核验。
4. **行数**：与冻结的输入 manifest 逐文件精确匹配。已知的上游固定计数包括 Stage-2C recovery 行 `17280`、Stage-2C event cohort `96`、replay-valid subset `63` 和 invalid event 数 `33`；任何预期计数不符均报告差异，不改数据凑数。
5. **唯一键**：按文件声明的唯一键逐行检查，不允许重复、缺失、隐式覆盖或静默去重。
6. **必需字段**：字段缺失、null、NaN、无穷、非法枚举、错误 split、错误 cohort 标记均为前置失败。
7. **官方 sharded loader**：2D 的 sharded pointer 只能通过项目已冻结的官方 `load_jsonl(pointer)` 读取；禁止自行 glob、手工拼接、按文件名推断 shard 顺序或静默忽略缺片。
8. **任务隔离**：若一个任务缺字段或前置失败，只将该任务标记 `BLOCKED`，其他任务继续；不得用其他任务数据填补。
9. **冻结报告一致性**：若重现既有冻结报告的权威计数不一致，停止相应任务并原样报告差异；禁止修改 artifact 或脚本去对齐。
10. **失败传播**：任一 upstream failure 后不得产生正面标签、G0 PASS 或选择建议；只能输出 `BLOCKED`/`INCONCLUSIVE`/待测。

所有输出 JSON 使用稳定 key 顺序和稳定记录排序；所有浮点判定容差固定为 `1e-12`。后续所有 Stage-2E 产物完成后，才由单独步骤生成覆盖全产物的 `artifacts/stage2e/SHA256SUMS`。

## 3. 固定 cohort、定义与标签

### 3.1 Stage-2C 双 cohort

任何基于 Stage-2C 的数字必须同时产生两个版本，并显式标记污染状态：

- `all_contaminated`：完整 96-event cohort，包含已知 shared mutable runtime 污染。
- `replay_valid_subset`：排除 `invalid_events` 后的 63-event subset。该版本只能称为 `shared_runtime_no_detected_violation`，不得称为 fresh-process clean、clean replay 或独立 runtime 证据。

所有双 cohort 表保留原始 split、task、parameter、operator 和 actor seed；不得把两个 cohort 合并后只报有利版本。Stage-2D 如被用于描述，必须标记 `fresh_process_verified`，但整个 S0 仍然是 `diagnostic_only`。

### 3.2 safe success 与边界

沿用输入 artifact 已冻结的 `safe_success`、`cause_violation`、`S_obs` 和 recovery actor row 定义；S0 不重定义、不替换 cause，也不从标量 contact count 推导新 cause。

若规则产生 `boundary=None`，唯一 fallback 是 `prefix_k=d=2`，并使用该方法的 fresh-h16 recovery actor rows；不可删除该 event、不可用 evaluation 结果补齐、不可将 fallback 伪装为观察到的边界。

## 4. 任务 A：共同支撑重分析（Q1）

### 4.1 输入与分析分母

允许在后续执行阶段读取的冻结输入为：

```text
artifacts/stage2c/formal_matrix/recovery_operator_rows.jsonl
artifacts/stage2c/recoverability_atlas/rows.jsonl
artifacts/stage2c/recoverability_atlas/event_boundaries.jsonl
artifacts/stage2c/recoverability_atlas/invalid_events.jsonl
artifacts/stage2c/crossfit_replanability/baseline_rows.jsonl
artifacts/stage2c/crossfit_replanability/crossfit_rows.jsonl
```

A1 只复现 Stage-2C 报告中权威的 evaluation/deployable baseline machine table，覆盖分母为 48 events、每 event 3 actor rows。A2–A5 的主分析只使用预先标记的 calibration 48 events；不得将 evaluation split 混入选择或 calibration 统计。Stage-2C 总 cohort 的 all/valid 双版本必须按第 3.1 节保留。

### 4.2 A1 权威复现核对（预期常数）

下表是**事前冻结的复现目标**，不是本提交产生的实验结果。执行时同时核对精确计数和一位小数显示；任何一项不符都令 `A1=BLOCKED`，只记录差异，不改数据。

| baseline | 预期分子/分母 | 预期百分比 | 预期 support coverage |
| --- | ---: | ---: | ---: |
| `action_disagreement` | `10/144` | `6.9%` | `48/48 events` |
| `fixed_delay_1` | `8/144` | `5.6%` | `48/48 events` |
| `fixed_delay_2` | `13/144` | `9.0%` | `48/48 events` |
| `fixed_delay_4` | `15/144` | `10.4%` | `48/48 events` |
| `fixed_delay_8` | `16/144` | `11.1%` | `48/48 events` |
| `immediate_fresh_h16` | `23/144` | `16.0%` | `48/48 events` |
| `k_last_observed_safe` | `3/72` | `4.2%` | `24/48 events` |
| `k_last_recoverable` | `20/54` | `37.0%` | `18/48 events` |
| `velocity_phase` | `16/144` | `11.1%` | `48/48 events` |

百分比先保留精确有理数，再以 `ROUND_HALF_UP` 显示一位小数；精确分子/分母和显示百分比都必须通过。support coverage 的 event 分母固定为 48，不能以 actor-row 分母替代。

### 4.3 A2/A3 全支撑重算与 strongest fixed delay

baseline 候选集合严格冻结为以下九项，不增加、不删减、不用诊断项替代：

```text
action_disagreement
fixed_delay_1
fixed_delay_2
fixed_delay_4
fixed_delay_8
immediate_fresh_h16
k_last_observed_safe
k_last_recoverable
velocity_phase
```

对每个 baseline 输出三列并排结果：

- `restricted_support`：原始 boundary 有定义的 event 子集，保留自然 support 分母；
- `full_support`：对 boundary 未定义 event 使用 `prefix_k=d=2` 的显式 fallback，重新覆盖全部 48 calibration events、144 actor rows；
- `support_coverage`：`defined_events/48`，并同时记录 actor-row coverage `defined_rows/144`。

fallback 的 recovery rows 必须来自该 method 对应的 fresh-h16 recovery rows；禁止用其他 method 的 outcome、手工常数、插值或删除未定义 event。

`strongest_fixed_delay` 只能在 calibration 中从严格集合 `{fixed_delay_1, fixed_delay_2, fixed_delay_4, fixed_delay_8}` 选择。排序规则固定为：

1. `safe_success` 降序；
2. `new_non_nominal_actions` 升序；
3. `actor_calls` 升序；
4. `method` 字典序升序。

不得用 evaluation outcome、CI、actual usage、后验机制解释或人工偏好破坏该排序。

`k_last_recoverable - strongest_fixed_delay` 的差值先在每个 event 内平均三条 recovery actor rows，再按 `cluster=(task, init_state_id)` 等权聚合。paired cluster bootstrap 固定为 10000 次、seed `216214`、有放回重采样 cluster、percentile 95% CI `[2.5, 97.5]`。prefix row 不是独立样本；actor checkpoint 是重复测量，须单独报出。

输出必须含 restricted/full/coverage 三个版本，并标注 `all_contaminated` 与 `replay_valid_subset`。

### 4.4 A4 共同支撑优势与选择效应

相对于同一个 calibration-only `strongest_fixed_delay`，冻结以下量：

```text
restricted_delta = restricted_kLR_safe_success - restricted_winner_safe_success
full_delta       = full_kLR_safe_success - full_winner_safe_success
absolute_inflation = restricted_delta - full_delta
explained_fraction = (restricted_delta - full_delta) / restricted_delta, 仅当 restricted_delta > 0
```

当 `restricted_delta <= 0` 时 `explained_fraction=NA`，不得设为 0。A4 还必须将既有 held-out Track-B 数字 `-0.09444444444444444` 并排展示；该常数只作为冻结的历史描述，不能被加载为当前选择信号，也不能参与任何 S0 选择。

### 4.5 A5 event-level cause 聚合

对每个 event 的全部 recovery rows（`15 k positions × 4 operators × 3 actor rows = 180 rows/event`）进行 event-level 聚合，不把 prefix rows 当独立事件：

```text
cause_violation_type_set = sorted(unique(non-empty cause_violation_type values))
prefix_cause_violation_any = any(cause_violation over the 180 rows)
prefix_cause_violation_fraction = mean(cause_violation over the 180 rows)
```

按 `parameter_id`、`task` 比较预先定义的 covered 21 events 与 uncovered 27 events，分别给出 cause 类型集合、`any`、fraction 和 support coverage；同时输出 `all_contaminated` 与 `replay_valid_subset` 两版。任何 invalid event 不得被静默视作安全或从全集中抹除。

### 4.6 G0-1：A 路径独立判决

设 `full_delta` 为 A4 的全支撑安全成功差，CI 为上述 cluster bootstrap CI；数值判定容差固定 `1e-12`：

- **(a) 全支撑保留优势**：`full_delta` 的 CI lower bound `> -0.05`，且优势未被全支撑/污染检查否定。G0-1 输出 `FULL_SUPPORT_ADVANTAGE`，S1 才可将 `k_last_recoverable` 作为候选 selector 测量对象。
- **(b) 优势归零**：`abs(full_delta) <= 1e-12`。G0-1 输出 `ZERO_FULL_SUPPORT_ADVANTAGE`，S1 放弃“打赢 baseline”主张，只能测描述性的 operator relativity。
- **(c) 优势反号**：`full_delta < -1e-12`。G0-1 输出 `NEGATIVE_FULL_SUPPORT_ADVANTAGE`，记录“保留安全前缀有增益”框架的负结果并等待人工决策；不自行进入 S1。
- 其他情况输出 `INCONCLUSIVE`。S1 只能按 B/C 的准备性结果推进，不得宣称 selector 证据。

G0-1 只决定 S1 应测的主张，不决定是否可以绕过其他 gate；它绝不与 G0-2、G0-3 合并总分。

## 5. 任务 B：头顶空间审计（Q2）

### 5.1 对齐轴与污染标记

B1 的唯一对齐表字段顺序冻结如下：

```text
stage
split
task
semantic_family
original_parameter
runtime_purity
event
prefix
operator_arm
tail_horizon
configured_action_budget
configured_call_budget
actual_usage
safe_success
```

规则：

- Stage-2C 产生 `all_contaminated` 与 `replay_valid_subset` 双版；Stage-2D 只标 `fresh_process_verified`，但 S0 总结仍标 `diagnostic_only`。
- 不跨 stage 做 event join；不把相同毫米位移 (`mm`) 视为相同 severity；不把 `actual_usage` 当作 configured budget。
- 由于缺少 factorial/common-support 设计，预算与 severity 的因果贡献百分比统一冻结为 `NOT_IDENTIFIABLE`。
- 只允许量化 Stage-2C `all_contaminated → replay_valid_subset` 的 runtime contamination sensitivity；budget、severity、cross-stage 差异仅并排描述，不可声称因果贡献。
- 禁止 Shapley、顺序替换或任何把不可识别的交叉差异伪装成因果百分比的做法。

Stage-2D evaluation loader 在 B2/B3 冻结前必须处于 `open-deny` 状态；B1 也不得读取 Stage-2D reserve 80–99 或把 evaluation outcome 用于候选选择。

### 5.2 B2 结构性工作区间候选

B2 只使用 calibration outcome。每个 stage 内先按 structural cell 分组，再在 event 内平均 recovery actor 重复测量，最后对独立 `cluster=(task, init_state_id)` 等权平均。定义：

```text
oracle_best = max(safe_success over comparable arms)
weakest_arm = min(safe_success over comparable arms)
gap = oracle_best - weakest_arm
```

一个 cell 只有同时满足以下条件才是候选：

- `oracle_best ∈ [0.25, 0.85]`（含端点）；
- `gap >= 0.15`；
- 至少 6 个独立 `(task, init_state_id)` clusters；
- 至少 2 个可比较 arms；
- 配置 action budget、tail horizon、configured policy-call budget 均被显式记录；
- 至少两个预设 configured budget levels 的受控 grid，而不是由 event 剩余 horizon 反推的伪 grid。

必须列出所有满足条件的 tuple：

```text
(configured_action_budget, tail_horizon, configured_call_budget,
 task, parameter_id, runtime_cohort)
```

Stage-2C 必须分别报告 all/valid 两版；G0-2 只允许使用 `replay_valid_subset` 或 Stage-2D fresh-process calibration 候选。若没有受控的至少两个 preset budget levels，即使有描述性候选也不能转成 B3 因果建议，G0-2 固定输出 `BUDGET_SCAN_FIRST`。

### 5.3 B3 S1 预算建议

只有 B2 存在合格且受控的候选时才输出一个具体预算 tuple，不输出区间。tie-break 固定为最小：

```text
(max_policy_calls,
 configured_action_budget,
 tail_horizon,
 canonical_severity_key,
 canonical_operator_key)
```

不得用 effect 大小、CI、actual usage、evaluation outcome 或事后成功率打破 tie。若没有合格候选，不编造数值，输出 `S1_FIRST_ACTION=BUDGET_SCAN_FIRST`，并明确 S1 先做预算扫描。

### 5.4 G0-2：B 路径独立判决

- B2 找到受控工作区间且 B3 能给出具体 tuple：`G0-2=CONTROLLED_HEADROOM_FOUND`。
- B2 找不到任一工作区间，或预算 grid 不受控：`G0-2=BUDGET_SCAN_FIRST`。
- B 路径不能因为 A 或 C 的结果改变自己的判决；不能从 G0 总分推导。

## 6. 任务 C：算子相对 recoverability 阶梯（Q3）

### 6.1 冻结 operator family 与 recovery 定义

从 Stage-2C atlas 行离线构造以下嵌套 family，不能重跑 runtime：

```text
U1 = {fresh_h4}
U2 = {fresh_h4, fresh_h16}
U3 = U2 ∪ {hold_one_step_then_fresh_h16}
U4 = U3 ∪ {rollback_one_step_then_fresh_h16}  (= 现有 U)
```

每个 prefix `k=2,3,...,16` 必须使用三条冻结 recovery actor checkpoints（seed 7、17、29）的完整 finite rows；每个 event 的 cell 必须有 `4 operators × 3 seeds`，否则该 event/cell `BLOCKED`，不得删行、补行或用均值替代缺失。

沿用以下定义：

```text
C_self(k) = 该 event 自身 actor checkpoint 的 safe-success
C_family(k) = 三个 recovery actor checkpoints 上的 safe-success fraction
C_family threshold = >= 2/3
R_U(k) = 1 iff 至少一个 operator in U 满足 C_family(k) >= 2/3，否则 0
k_last_recoverable(U) = max{k: R_U(k)=1}，不存在则 None
k_irrev_U = min{k: R_U(j)=0 对所有 j>=k}，不存在 persistent crossing 则 None
```

`k_irrev_U` 只表示 operator-relative 的 persistent crossing；不得把它用于任何超出本阶段范围的物理解释。

### 6.2 C1/C2 单调性审计

对每个 event 分别重算 U1、U2、U3、U4 的 `k_last_recoverable` 和 `k_irrev_U`；记录 raw nonmonotonic pattern，不得静默强制单调。逐 event 检查所有嵌套关系（U1⊂U2、U2⊂U3、U3⊂U4，并报告传递闭包）应满足：

```text
k_last_recoverable(Ui) <= k_last_recoverable(Uj)
```

在 C2 单调性审计中，`None` 仅用 sentinel `d-1=1` 比较；该 sentinel 不得进入 safe-success、G0、位移分布或 selector 选择。违反数不为 0 时，列出具体 event、operator family、k、原始 rows、finite/NaN 状态，定位非确定性、污染或缺失；不得平均掉违反。

### 6.3 C3/C4 位移与支撑统计

主位移只在 `k_last_recoverable(U4)` 与 `k_last_recoverable(U1)` **同时定义**的 event 上计算：

```text
absolute_shift = abs(k_last_recoverable(U4) - k_last_recoverable(U1))
```

对全集和 replay-valid subset 各自报告中位数、0/25/50/75/100 分位数及 `absolute_shift >= 1` 的 event 比例；分位数方法固定为 NumPy `linear`（等价 Type-7），不得改用 nearest/lower/higher。另报 U1→U4 support gains（每个 boundary 有定义的 event 数变化），不把缺失边界当作 0 位移。

若 U1 下 boundary 有定义的 event 少于 10（个位数），必须明确写明 operator ladder 统计上不成立；不得用 U4 或 fallback 伪造 U1 支撑。

全集与 valid 双版本必须保留；任何双版本方向不一致都使 C 路径不能给出正向建议。

### 6.4 C5 机制反解与 S2 建议

只根据真实已读代码和原始行字段反解各 operator 的 reach 差异：

- 说明 fresh、hold、rollback 的行为定义、调用时机、action/call budget 和 outcome 观测方式；
- 将 reach 差异与 C3/C4 的实际边界变化逐项对应；
- 若缺少某类 operator，只给出 S2 所需的**行为定义**（输入、动作语义、预算、输出），不实现、不运行；
- 不生成新的 idea，`new_idea_generated=false`；不把机制说明改写成部署主张。

### 6.5 G0-3：C 路径独立判决

当前 operator ladder 只有在下列全部条件同时满足时可用：

- 两个 boundary 同时定义的 event 中，`absolute_shift >= 1` 的比例 `>=30%`；
- 单调性 violation 数为 0；
- U1 defined event 数 `>=10`；
- `all_contaminated` 与 `replay_valid_subset` 的方向一致。

满足时 `G0-3=OPERATOR_LADDER_USABLE`；否则 `G0-3=EXPAND_OPERATOR_FAMILY_FIRST`，S2 必须先按 C5 扩宽 operator family。若单调性 violation 非零，说明 R_U 存在非确定性、污染残留、缺失或 NaN；因为集合嵌套，理论上的单调性本身恒真，不能把 violation 直接解释成污染证据或平均掉。

## 7. S1 仪器化契约（冻结规范，暂不实现）

S1 如获准，必须另提交 `experiments/r16_p14_stage2e/INSTRUMENTATION_S1.md` 并按以下不可变契约记录。S0 不运行 runtime，不生成实现：

1. 每一步记录完整接触对集合（规范化的 geom name pair 列表），不能只记 `contact_count`。
2. 行记录保留原始接触拓扑和 raw geometry；cause 判据由可替换的离线纯函数计算，cause label 不写死在 runtime。
3. 同时记录一个 release 之前可触发的候选量（例如被操作物与非目标 geom 的接触拓扑越界），并与现有 release-based cause 并存；不能替换或覆盖 release cause。
4. 每条 branch 记录 process ID、environment construction hash、chunk bytes hash，并沿用 Stage-2D isolation signature 字段格式。
5. 每步保留 `pid/env_hash/chunk_hash` 与对应 source event、prefix、operator、actor seed 的唯一键；缺失任一签名时该 branch/event blocked。

## 8. 结果、G0 输出和负责任边界

### 8.1 三路独立输出

后续 S0 结果必须分别输出 A、B、C 的 summary、数据覆盖、阻塞原因、G0 判决和待测项。方法性输出统一含：

```text
diagnostic_only=true
formal_positive_evidence_allowed=false
new_idea_generated=false
planned_pai_jobs=0
submitted_pai_jobs=0
```

任何一个任务的字段缺失、行数错误、SHA 不符、split 污染、报告复现差异或 upstream failure，只阻塞该任务；受阻任务不产生 positive label。不存在“总分过 gate”或用另一条路径补救的规则。

### 8.2 停止/待测

- 需要 GPU、模型、仿真或 rollout：停止该问题，记入 S1 待测。
- 冻结 artifact 与既有报告不一致：停止相应任务，报告 exact diff，不修 artifact。
- A1 不一致：A `BLOCKED`，不修改数据。
- C1 cell 不完整、NaN、缺 actor seed：该 event/cell `BLOCKED`，不补齐。
- 完成 S0 后不要自行开启 S1；S0 只给下一步测量准备结论。

### 8.3 术语与结论边界

报告只能把边界称为 operator-relative、observed、policy-relative 或 recoverability 统计，不把它升级为普适或物理层结论。不得声称：

- general R16-P14 validation；
- VLA validation、训练成功或 learned intervention policy；
- accepted idea 或高于 `N2` 的 novelty；
- 任何依赖本 S0 的 deployable/causal 正面证据。

novelty ceiling 继续固定为 `N2_ORACLE_PROTOCOL_BOUNDARY_ONLY`，`accepted=false`。

## 9. 后续执行与提交闸门

本 S0 文件必须先于所有 Stage-2E 分析提交，并作为后续分析的 parent hash 输入。后续执行者必须：

1. 从本文件记录的 immutable parent/branch/worktree 重新核对 Git tree；
2. 在 source freeze 中保存所有实际输入 SHA、容器/schema/行数/唯一键校验；
3. 只写 Stage-2E 新目录，保留 all/valid 双 cohort；
4. 生成每个产物的 SHA256，并在末次提交前验证 `SHA256SUMS` 覆盖全集；
5. 在 report 中明确区分历史 immutable facts、diagnostic reanalysis、污染敏感性和待测内容；
6. 不自动创建 PAI job，不自动启动 S1。

本首提交的验收条件是：

```text
git diff edebdfc64576129d994535dacb76de930f493c8d..HEAD --name-only
=> experiments/r16_p14_stage2e/PREREG_S0.md
```

除上述单一文件外，本提交不得包含任何代码、数据、结果、报告、脚本、模型或 PAI 记录。
