# Phase0A 机制审计

本审计只覆盖已经完成的 Phase0A。输入是四个冻结的 Stage2C artifact：`c_recovery`、`c_boundaries`、`c_invalid` 和 `c_baseline`。`phase0a.py:run()` 先按冻结 SHA256 校验并调用冻结脚本的校验器，再通过 AST 适配 `_fixed_winner` 和输出目录；它没有读取 S1 calibration/evaluation，也没有重新运行仿真。`summary.json` 记录了 `source_inputs.scope = "only four frozen Stage2C A inputs; no S1 outcomes"`、`simulation_steps = 0`、`gpu_hours = 0.0`、`new_idea_generated = false`。

## 可复现性边界

代码复现门通过了，但结论仍然是诊断性的：`summary.json` 的 `A1_exact_reproduction` 为 `true`，`status` 和 `G0_1` 都是 `INCONCLUSIVE`，`positive_label_allowed`、`formal_positive_evidence_allowed` 和 `performance_advantage_claimed` 都为 `false`。A1 的九个冻结计数可直接在 `summary.json:A1_rows` 复核：`action_disagreement=10/144 (6.9%)`、`fixed_delay_1=8/144 (5.6%)`、`fixed_delay_2=13/144 (9.0%)`、`fixed_delay_4=15/144 (10.4%)`、`fixed_delay_8=16/144 (11.1%)`、`immediate_fresh_h16=23/144 (16.0%)`、`k_last_observed_safe=3/72 (4.2%)`、`k_last_recoverable=20/54 (37.0%)`、`velocity_phase=16/144 (11.1%)`。这些是冻结 artifact 的复算检查，不是 S1 结果。

`phase0a.py:adapted_module()` 只做两件事：把 `immediate_fresh_h16` 加入冻结 `_fixed_winner` 的候选集合，并把 A 输出重定向到 `artifacts/stage2f/phase0a`。其余方法、输入校验、均值和 bootstrap 来自 `scripts/run_r16p14_stage2e_s0.py`。因此下面解释的是已运行代码实际形成的估计量，而不是把表格当成独立的 actor 观测。

## 38.1% 到 19.4% 的机制

这个变化出现在 `table.csv` 的 `all_contaminated,k_last_recoverable` 行：

| 支持定义 | 数值 | 直接计数 | 事件数 | 代码路径 |
|---|---:|---:|---:|---|
| restricted | `0.38095238095238093 = 24/63` | 24 个成功 actor rows / 63 个 actor rows | 21 个事件 | `_method_summary(..., restricted=True)` |
| full | `0.19444444444444445 = 28/144` | 28 个成功 actor rows / 144 个 actor rows | 48 个事件 | `_method_summary(..., restricted=False)` |

这里每个已定义事件恰有三个 actor rows，并且 `_summarise_method_event_rows()` 先对三个 actor 取事件均值，`_method_summary()` 再对事件均值取 `mean()`。由于每个事件都是三个 actor，这两个分数也可以用表中的整数 actor-row 计数复核；正式含义仍是 **event mean of three-actor means**。

分母改变来自支持规则，而不是模型重新运行。冻结脚本的 `_event_method_rows()` 对 `k_last_recoverable` 先寻找该事件的 baseline rows：

1. `restricted=True` 时，没有原始 `k_last_recoverable` rows 的事件直接不进入分母。因此只剩 21 个事件、63 个 actor rows。
2. `restricted=False` 时，对这些未定义事件使用唯一允许的 fallback：`_fresh_fallback_rows()` 从 `c_recovery` 取 `prefix_k=2`、`operator="fresh_h16"` 的三个 actor rows。于是增加了 27 个事件、81 个 actor rows。

`support_decomposition.json` 给出了同一变化的整数审计：`all_contaminated` 的 restricted 是 `24/63`，full 是 `28/144`，新增 fallback 是 `4/81`。所以

```text
(24 + 4) / (63 + 81) = 28 / 144 = 0.19444444444444445
```

大量新增 fallback rows 只有 4 个成功，令估计量从 38.0952% 变成 19.4444%。这表示 **k_last_recoverable 在 method-defined support 上的高值没有外推到全部 48 个事件**；它没有表示模型在相同事件、相同干预下从 38.1% 退化到了 19.4%。两种支持还使用了不同的事件集合，因此不能把两者的差直接当作同一批事件的 treatment effect。

相同机制也出现在 `replay_valid_subset`，可作独立的 artifact 复核：`k_last_recoverable` 从 restricted `12/27=0.4444444444444444` 变为 full `15/102=0.14705882352941174`；新增 fallback 是 `3/75`，新增事件数为 25。它说明下降随支持扩张出现，而不是 `all_contaminated` 某一个小数的四舍五入。

## 加入 immediate 后相对优势为何弱化

冻结脚本的 `_fixed_winner()` 原本只在四个 `fixed_delay_*` 之间按 `safe_success` 降序选择；`phase0a.py:adapted_module()` 把 `immediate_fresh_h16` 加入同一候选集合后，仍按同一排序键选择 winner。这个改动不改变任何 `k_last_recoverable` row，只改变参照 arm。

`table.csv` 的 full-support event means 可以直接列成 actor-row 分数：

| cohort | fixed_delay_8 | immediate_fresh_h16 | k_last_recoverable | kLR − fixed_delay_8 | kLR − immediate |
|---|---:|---:|---:|---:|---:|
| all_contaminated | `17/144=0.1180556` | `24/144=0.1666667` | `28/144=0.1944444` | `11/144=0.0763889` | `4/144=0.0277778` |
| replay_valid_subset | `8/102=0.0784314` | `13/102=0.1274510` | `15/102=0.1470588` | `7/102=0.0686275` | `2/102=0.0196078` |

在四个 fixed-delay 候选中，`fixed_delay_8` 是最高者；加入 immediate 后，参照值从 11.8056% 提高到 16.6667%，因此同一 full-support kLR 的表面差距从 7.6389 个百分点缩到 2.7778 个百分点。valid subset 的差距也从 6.8627 个百分点缩到 1.9608 个百分点。这里的“弱化”是候选集合扩展导致的 comparator replacement，可以由 `phase0a.py:adapter_changes` 和 `summary.json:cohorts.*.strongest_fixed_delay = "immediate_fresh_h16"` 直接复核；它不是模型性能下降。

## event mean 与 cluster mean 必须分开

`table.csv` 和 `method_summaries` 的 safe-success 是 event mean：每个事件先平均三个 actor，再对事件等权。`summary.json:absolute_comparisons_same_support` 另外报告了 preregistered cluster bootstrap 的估计量；`run_r16p14_stage2e_s0.py:cluster_bootstrap()` 按 `(task,init_state_id)` 聚簇，先在簇内平均事件值，再对簇等权。它不是把所有 actor rows 或所有事件直接池化。

对 `all_contaminated`，full support 恰好是 16 个 cluster、48 个事件，分布使 cluster mean 与 event mean 都为 kLR `0.1944444`、immediate `0.1666667`，paired cluster delta 为 `+0.0277778`，但 10000 次 bootstrap 的 95% CI 是 `[-0.0555556, 0.125]`。它跨过 0，不能给出正式的正向证据。

restricted support 只有 12 个 cluster、21 个事件，事件并非每个 cluster 数量相同，所以两个估计量分开：

- event mean：kLR `24/63=0.3809524`；同一事件集合上的 immediate 为 `0.3174603`，这是描述性 pooled event mean。
- cluster mean：kLR `0.3472222`，immediate `0.3657407`，paired cluster delta `-0.0185185`；bootstrap 95% CI 为 `[-0.25, 0.1759259]`。

因此不能拿 restricted 的 `38.1%` 与 full 的 `19.4%`，或者拿 restricted event-level 的正差，与 cluster-level 的负差混成一个方向结论。summary 中的 `absolute_inflation=-0.0462963` 只是 restricted cluster delta 减去 full cluster delta；`explained_fraction` 为 `null`，它没有被解释成因果比例。

`replay_valid_subset` 的 cluster 结果同样不支持正向结论：restricted delta `-0.0740741`、CI `[-0.5,0.2962963]`；full delta `+0.0158730`、CI `[-0.0634921,0.1190476]`。两者都跨过 0。

## 能推出与不能推出的结论

可以推出的是：在冻结 Stage2C artifact 和当前代码的定义下，`k_last_recoverable` 的 restricted 高值主要依赖 method-defined support；把未定义事件纳入 full support 时，唯一允许的 fresh-h16-at-k2 fallback 以低成功率补入分母。加入 `immediate_fresh_h16` 后，固定延迟参照 arm 被更高的 immediate 值替换，kLR 相对差距相应缩小。A1 原始计数可复现，且所有这些比较都被标记为 diagnostic-only。

不能推出的是：模型发生了 38.1% 到 19.4% 的真实退化；kLR 对 immediate 或 fixed-delay 的因果优越性；fallback 支持扩张对所有新事件的无偏效果；以及对 S1 calibration/evaluation、机器人运行或 PAI 任务的外推。`G0_1=INCONCLUSIVE`、`formal_positive_evidence_allowed=false`、`performance_advantage_claimed=false` 和 `planned_pai_jobs=0` 是本阶段的最终边界。

本报告没有提出新 idea 或新增实验，只按 code-first 机制反解已完成 Phase0A 的实际代码、分母和估计量。

## 增补：A1 原始计数与 Phase0A 表格的逐行对账

父验收发现，A1 的冻结 evaluation 原始计数是 `immediate_fresh_h16=23/144`、`fixed_delay_8=16/144`，而 Phase0A `table.csv` 的 full-support 行是 `24/144`、`17/144`。这不是四舍五入，也不是 fallback 或 cluster 重加权造成的 +1；两处读取的是同一个 `c_baseline` artifact 的不同 split。

代码路径可以逐行定位这个分叉。`experiments/r16_p14_stage2f/phase0a.py:33-37` 只读并校验四个冻结输入：`c_recovery`、`c_boundaries`、`c_invalid`、`c_baseline`。随后 `phase0a.py:46` 明确构造 `calibration = c_baseline[split == "calibration"]`，并在 `phase0a.py:52-55` 用这批 calibration rows 调用 `_method_summary`；因此它写入的 `table.csv` 是 calibration support。冻结 `scripts/run_r16p14_stage2e_s0.py:638-645` 则明确构造 `eval_baseline = baseline[split == "evaluation"]`，A1 对该集合计数。故两者的分母都为 48 events × 3 heldout actors = 144 rows，但事件集合不是同一批：该 artifact 中 calibration 行的 `init_state_id` 为 30–37，evaluation 行为 40–48；没有同一 `event_instance_id` 的 calibration/evaluation 配对。

| method | `table.csv` 使用的 split | table raw true rows | A1 使用的 split | A1 raw true rows | 净差 |
|---|---|---:|---|---:|---:|
| `fixed_delay_8` | calibration | 17/144 = 11.8055556% | evaluation | 16/144 = 11.1111111% | +1 row |
| `immediate_fresh_h16` | calibration | 24/144 = 16.6666667% | evaluation | 23/144 = 15.9722222% | +1 row |

这些数值可由 `artifacts/stage2f/phase0a/a1_published_evaluation.csv`、`artifacts/stage2f/phase0a/table.csv` 直接回读，并由 `c_baseline` 的逐行过滤复核。校验后的四个输入行数和 SHA256 仍为：`c_recovery=17280/21bd70d4dd258dd4e4b52a453ca137b2552800a40654bb03db0bbdb344e47efa`，`c_boundaries=96/d2c93d6027419a850220b1c1fc644579d4f1387633099c34129a9643305c7ff9`，`c_invalid=33/eae0ce6f2804a3628c46bdd55d1f5abcae6b16dab35f8e194914f9fd9ec02f13`，`c_baseline=2289/2d212f4a7f0a4062866feed39482f67a9622cd2b8aaaabff332874552d3a7626`。四者均只读，未重新运行 rollout。

以下是 `c_baseline` 中所有 `safe_success=true` 的受影响原始行。每项格式为 `event_instance_id : heldout_actor_seed`；同一 method/split 下未列出的行均为 `safe_success=false`。这给出逐 actor 行证据，而非仅给出聚合比例。

### `fixed_delay_8`

- calibration（17 rows，11 events；task 分解：stove 9、cream-cheese 8）：
  - `put_the_bowl_on_the_stove__seed07__init31__future_06_lateral_040mm:7`
  - `put_the_bowl_on_the_stove__seed07__init36__future_14_lateral_020mm:17`
  - `put_the_bowl_on_the_stove__seed17__init33__future_14_lateral_020mm:17,29`
  - `put_the_bowl_on_the_stove__seed17__init37__future_14_lateral_020mm:29`
  - `put_the_bowl_on_the_stove__seed29__init34__future_14_lateral_020mm:7,17,29`
  - `put_the_bowl_on_the_stove__seed29__init36__future_14_lateral_020mm:17`
  - `put_the_cream_cheese_in_the_bowl__seed17__init31__shift_040mm:7`
  - `put_the_cream_cheese_in_the_bowl__seed17__init35__shift_040mm:7,17,29`
  - `put_the_cream_cheese_in_the_bowl__seed17__init36__shift_060mm:29`
  - `put_the_cream_cheese_in_the_bowl__seed17__init37__shift_040mm:7`
  - `put_the_cream_cheese_in_the_bowl__seed29__init34__shift_040mm:17,29`
- evaluation（16 rows，11 events；task 分解：stove 12、cream-cheese 4）：
  - `put_the_bowl_on_the_stove__seed07__init42__future_14_lateral_020mm:17`
  - `put_the_bowl_on_the_stove__seed07__init44__future_14_lateral_020mm:7,29`
  - `put_the_bowl_on_the_stove__seed07__init47__future_06_lateral_040mm:29`
  - `put_the_bowl_on_the_stove__seed17__init42__future_06_lateral_040mm:29`
  - `put_the_bowl_on_the_stove__seed17__init46__future_06_lateral_040mm:7,17`
  - `put_the_bowl_on_the_stove__seed29__init40__future_14_lateral_020mm:7,29`
  - `put_the_bowl_on_the_stove__seed29__init41__future_06_lateral_040mm:17`
  - `put_the_bowl_on_the_stove__seed29__init42__future_14_lateral_020mm:7,17`
  - `put_the_cream_cheese_in_the_bowl__seed17__init45__shift_040mm:7`
  - `put_the_cream_cheese_in_the_bowl__seed29__init42__shift_040mm:29`
  - `put_the_cream_cheese_in_the_bowl__seed29__init45__shift_060mm:17,29`

### `immediate_fresh_h16`

- calibration（24 rows，15 events；task 分解：stove 15、cream-cheese 9）：
  - `put_the_bowl_on_the_stove__seed07__init34__future_14_lateral_020mm:7`
  - `put_the_bowl_on_the_stove__seed07__init35__future_06_lateral_040mm:7`
  - `put_the_bowl_on_the_stove__seed17__init33__future_14_lateral_020mm:7,29`
  - `put_the_bowl_on_the_stove__seed17__init34__future_06_lateral_040mm:17,29`
  - `put_the_bowl_on_the_stove__seed17__init35__future_14_lateral_020mm:7`
  - `put_the_bowl_on_the_stove__seed17__init36__future_06_lateral_040mm:7`
  - `put_the_bowl_on_the_stove__seed17__init37__future_14_lateral_020mm:7,17`
  - `put_the_bowl_on_the_stove__seed29__init32__future_14_lateral_020mm:7,29`
  - `put_the_bowl_on_the_stove__seed29__init34__future_14_lateral_020mm:7,17,29`
  - `put_the_cream_cheese_in_the_bowl__seed17__init31__shift_040mm:17`
  - `put_the_cream_cheese_in_the_bowl__seed17__init35__shift_040mm:29`
  - `put_the_cream_cheese_in_the_bowl__seed17__init36__shift_060mm:7,29`
  - `put_the_cream_cheese_in_the_bowl__seed17__init37__shift_040mm:7,17,29`
  - `put_the_cream_cheese_in_the_bowl__seed29__init34__shift_040mm:17`
  - `put_the_cream_cheese_in_the_bowl__seed29__init36__shift_040mm:7`
- evaluation（23 rows，14 events；task 分解：stove 20、cream-cheese 3）：
  - `put_the_bowl_on_the_stove__seed07__init41__future_06_lateral_040mm:7`
  - `put_the_bowl_on_the_stove__seed07__init42__future_14_lateral_020mm:7,17,29`
  - `put_the_bowl_on_the_stove__seed07__init44__future_14_lateral_020mm:7,29`
  - `put_the_bowl_on_the_stove__seed07__init45__future_06_lateral_040mm:7,17`
  - `put_the_bowl_on_the_stove__seed07__init47__future_06_lateral_040mm:7`
  - `put_the_bowl_on_the_stove__seed17__init42__future_06_lateral_040mm:7,17,29`
  - `put_the_bowl_on_the_stove__seed17__init44__future_06_lateral_040mm:29`
  - `put_the_bowl_on_the_stove__seed17__init46__future_06_lateral_040mm:7,17,29`
  - `put_the_bowl_on_the_stove__seed29__init41__future_06_lateral_040mm:7,17`
  - `put_the_bowl_on_the_stove__seed29__init42__future_14_lateral_020mm:29`
  - `put_the_bowl_on_the_stove__seed29__init44__future_14_lateral_020mm:7`
  - `put_the_cream_cheese_in_the_bowl__seed07__init40__shift_040mm:29`
  - `put_the_cream_cheese_in_the_bowl__seed17__init45__shift_040mm:17`
  - `put_the_cream_cheese_in_the_bowl__seed29__init45__shift_060mm:7`

For these two methods the frozen `_event_method_rows` path is direct baseline lookup (`run_r16p14_stage2e_s0.py:518-523`, `536-549`): `c_recovery` fallback is only reachable for `k_last_observed_safe` / `k_last_recoverable`, so it cannot account for either +1. Both methods have all 48 calibration events defined and `fallback_event_count=0`; restricted and full table rows therefore coincide. The table statistic remains event mean after three-actor mean, which here equals the 144-row true count divided by 144 because every event has exactly three actors. No reweighting can turn 23 into 24 or 16 into 17; the +1 is solely the calibration-versus-evaluation split switch.

The affected counts are therefore a split-support bookkeeping discrepancy, not a model change and not a claim about fallback behavior. The earlier report's substantive mechanism conclusion remains unchanged: it must keep A1's evaluation numbers (`23/144`, `16/144`) separate from calibration table numbers (`24/144`, `17/144`), and neither pair supports a causal performance claim.
