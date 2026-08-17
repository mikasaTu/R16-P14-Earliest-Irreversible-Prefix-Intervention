# R16-P14 Stage-2C：Recoverability-Defined and Compute-Matched Validation

## 一句话结论

最终决策是 **BLOCKED_BY_REPLAY_CONTRACT**：正式矩阵全部完成，但 replay contract 在聚合时因同一缓存 prefix 的 `S_obs(k)` 自相矛盾而 fail closed。Track A 为 **INCONCLUSIVE**，Track B 为 **INCONCLUSIVE**。这不是 gate 失败后的提前停止；`accepted=false`，新颖性上限仍为 `N2_ORACLE_PROTOCOL_BOUNDARY_ONLY`。

资格 gate 在查看方法收益之前已失败，正式矩阵又暴露了分支复放污染，因此后续数值不能获得正向标签，也不能被解释为干净的 cached/fresh 因果差异。所有 arm、统计和正负分组仍完整报告，用于定位实现机理与失败边界；任何结果都不称为物理不可逆性。

## 不可更改的历史结论

- Stage-1b universal hypothesis：`KILLED_IMMUTABLE`。
- Stage-2B chunk executability：`PASS`；`H_valid=16`。
- Stage-2B actor-conditioned perturbation 与 actor-history replay：`BLOCKED`。
- Stage-2B Track A/B：`INCONCLUSIVE`。
- Stage-2B local repair：`NO_SIGNAL`；operator router：`NO_OPERATOR_ROUTING_SIGNAL`。
- 本轮不修改、不覆盖、也不重新解释上述结论。

## 预注册、来源与算力

- 不可变父提交：`6eae66d23313cc97231249bfa1c40dc1767ea727`；tree：`ddcedd60f4f4e2878f8a4400d65e9e888f00cdd1`。
- PAI 正式来源提交：`55c180bfd1abb9767e146cbd7ec5554e6760a0d5`；tree：`a1e2caee307b2270bc167f26164d473e7908baf0`。
- 正式 PAI 作业：`dlcb8djiituf7gt3`，状态 `Succeeded`，时长 `56098` 秒。
- 正式推理使用 2×A800、两个隔离 GPU worker；没有训练 actor、没有 W&B、没有 π0.5、RGB/VLA、world model 或 learned head。
- PAI Python 为 `/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/libero_sft/bin/python`（SHA256 `89b2f5166fb529c259aedd43e5f718c60e35d58e630cb40ae6accb48fc4f961a`），overlay manifest SHA256 为 `64dfffdaf464d1a37be19b038cca919a252dba573eb2d0f8aa442b91a4099459`，正式 torch 为 `2.5.1+cu124`；launcher SHA256 为 `d3091f960ccb5626374bb60df4ebe81f1e628a79076078a49c7fca47f4db10f2`。
- 三个冻结 ACT checkpoint：`seed 7=821177a82cc470e108082fd3c0f6913983236a2fdf142de2fe51fc37c44240ca, seed 17=83ee61e31ffdae6f2ef57203a2c0085df41e284039cd81c6ecf1210694521604, seed 29=0cf34a3e535525345306a2b322aae3b1bd6ebd6cd71dc653e2a91393e2b79d1a`。所有分支使用冻结 actor、相同任务 init、相同扰动与预算合同。
- 重用的 240 个 immutable actor-event shards 清单 SHA256 为 `4fe57054d32c439201b81da0e433c9c0566c3222b21e28f994f584724474697c`。
- PAI lifecycle 收尾：`COMPLETE`；v3–v7 共 `5` 条 superseded Failed/Stopped service records 已逐条删除并验证缺席，v8 与成功 replay probe 保留，CPFS/registry 证据未删除。

## Phase A：正确性与合同修复

历史唯一 replay failure 首次可观察分歧出现在全局 step 181。四帧历史最大差异为 `1.862645149230957e-09`，模拟器状态最大差异为 `5.0034199006177005e-11`，ACT chunk 最大差异为 `6.4730644226074219e-05`。分类保持为 `LONG_HORIZON_NUMERICAL_CONTEXT_DRIFT_AMPLIFIED_BY_ACT`。

另外发现 Stage-2C 诊断 trace 曾调用 `current_observation`，从而刷新 LIBERO observable cache。正式提交把 trace=true/false 的刷新日程改成完全一致；独立 PAI probe 的三个新进程均精确重建。没有放宽容差。旧 Stage-2B stove replay 重新按全部尝试计分为 23/24，而不是丢弃错误后的 1.0。

其他修复包括：错误进入 replay 分母且 formal cell 要求 `error_count=0`；`S_obs(k)` 吸收单调，任何 0→1 使事件无效；任务进度来自 live BDDL predicate/site/object geometry，不再使用 demo-0 终点。以上修复通过了进入正式矩阵前的合同测试，但没有预先证明单个 EventRuntime 内连续分支 reset 的完整性。

## Phase B：扰动资格

资格尝试 `891/891`，missing=0，errors=0。最终状态 **BLOCKED**，标签 `BLOCKED_BY_SECOND_FAILURE_FAMILY`。

| 任务 | cells | 合格 cells | 注入无接触 cells | 最大 delayed violation | 失败检查计数 |
|-|-|-|-|-|-|
| open_the_top_drawer_and_put_the_bowl_inside | 9 | 0 | 3 | 7.1% | delayed_in_0_30_0_80=9, injection_contact_count_eq_0=6, interior_fraction_ge_0_20=2, median_offset_ge_2=3 |
| push_the_plate_to_the_front_of_the_stove | 9 | 0 | 0 | 0.0% | delayed_in_0_30_0_80=9, immediate_le_0_10=9, injection_contact_count_eq_0=9, interior_fraction_ge_0_20=9, median_offset_ge_2=9, two_actor_seeds_ge_5=9, valid_events_ge_20=9 |
| put_the_bowl_on_the_stove | 9 | 0 | 0 | 9.1% | delayed_in_0_30_0_80=9, injection_contact_count_eq_0=9, interior_fraction_ge_0_20=6, median_offset_ge_2=9 |
| put_the_cream_cheese_in_the_bowl | 3 | 0 | 3 | 0.0% | delayed_in_0_30_0_80=3, interior_fraction_ge_0_20=3, median_offset_ge_2=3 |

关键代码路径反解：target-shift 的 cause tracker 只有在物体已经 lift 后、gripper 从正变负时才会触发。冻结 calibration chunk 中仅 `3/30` 有检测后的释放，cause-eligible 为 `0`，因此 16-action 窗口内改变 shift 大小也无法产生预注册所需的 delayed violation。路径障碍则主要被注入时接触 gate 或 delayed violation 过低挡住。这解释冻结实现，不提出新机制。

用户明确要求 gate 失败后仍完成全部实验；因此下面的正式矩阵完整执行，但只作为 diagnostic，且 positive label 被 fail-closed 逻辑禁止。

## Phase C/D：事件池与完整执行

ACT 事件尝试 240，admitted=238，不合格排除=2，replay-unstable 排除=0。冻结 formal pool 为 96 个实例，minimum-data shape gate=`True`。

正式矩阵共有 96/96 个事件文件；matched rows=5760，recovery rows=17280，errors=0，`all_complete=True`。每个事件固定 60 条 matched-prefix 与 180 条 recovery-operator 记录。

预注册的主比较在代码参数上匹配：两支在 k 前都调用一次 detection-time ACT、都执行恰好 `k-d` 个 prefix action、都在 k 再调用 ACT、都使用 h=16 tail、相同动作上限/任务 horizon/seed/扰动/generator/evaluation actor。计划中的唯一处理差异是执行旧 chunk 的 `A[d:k]` 还是 detection 时新生成 chunk 的前 `k-d` 动作；但下面的正式 replay 审计证明运行态没有实现所需的分支隔离，因此不能把观测差异归因给这一个处理变量。

## 正式 replay contract：完整执行但 fail closed

恢复矩阵的 `1440` 个 `(event,k)` cell 每个都有 12/12 条预期记录；其中 `363` 个 cell 同时出现 `S_obs=true` 和 `false`，影响 `33/96` 个事件。分布为 `{'put_the_bowl_on_the_stove/calibration': 14, 'put_the_bowl_on_the_stove/evaluation': 19}`。cream-cheese 事件为 0 个，stove calibration/evaluation 分别为 14/19 个。

更强的同动作控制也失败：`CACHED_MATCHED` 和 `CACHED_NOQUERY` 在 k 前都执行相同冻结 `A[d:k]`，但 pre-tail `S_obs(k)` 在 `62/1440` 个 cell 不一致。恢复路径的 `prefix_cause_violation` 在 recovery actor 推理和 operator action **之前**写入；所以 actor、operator、tail outcome 和缺失行都不能解释这种差异。

代码级边界是：`run_event` 为一个事件只建立一个可变 LIBERO `EventRuntime`，然后按固定顺序运行所有 branch；`reset()` 只恢复枚举的 simulator/controller/RNG 状态，而不是为每个 branch 新建环境。原始证据因此把原因定位到 reset/detection/injection/prefix 路径中的残余可变状态或顺序依赖，但不足以唯一指定某个隐藏 simulator、controller 或 observable-cache 字段。没有修改原始 row，也没有用后处理补齐标签。

聚合文件中的 `INCOMPLETE_PREFIX_GRID` 不是少跑了 prefix：矛盾的 `S_obs` 集合被置为 null，随后 fail-closed 成 incomplete。正式执行仍是 96/96、23,040 rows、0 errors；科学 replay contract 则是 `BLOCKED`。这使全部 arm 数值、recoverability boundary、Track A/B 和正负机理分组只能作为污染数据上的描述性审计。

## 所有命名 arm（evaluation）

| arm | rows | safe success | task success | cause violation | 新动作 | actor calls | 完成步数 |
|-|-|-|-|-|-|-|-|
| cached_matched_all_prefixes | 720 | 14.2% | 15.8% | 58.6% | 221.69 | 17.71 | 230.69 |
| cached_noquery_all_prefixes | 720 | 11.9% | 14.0% | 62.6% | 226.53 | 16.71 | 235.53 |
| fixed_delay_1 | 48 | 4.2% | 6.2% | 68.8% | 251.40 | 17.98 | 254.40 |
| fixed_delay_2 | 48 | 6.2% | 8.3% | 68.8% | 245.25 | 17.96 | 249.25 |
| fixed_delay_4 | 48 | 10.4% | 10.4% | 62.5% | 237.17 | 17.83 | 243.17 |
| fixed_delay_8 | 48 | 8.3% | 10.4% | 64.6% | 233.46 | 17.65 | 243.46 |
| fresh_matched_all_prefixes | 720 | 11.9% | 13.6% | 61.5% | 233.96 | 17.71 | 235.96 |
| full_old_chunk | 48 | 10.4% | 10.4% | 62.5% | 226.85 | 17.12 | 242.85 |
| full_old_chunk_noquery | 48 | 8.3% | 10.4% | 66.7% | 230.52 | 16.12 | 246.52 |
| hold_prefix_matched_control | 720 | 9.6% | 12.5% | 63.5% | 239.65 | 17.71 | 241.65 |
| immediate_fresh_h16 | 48 | 16.7% | 22.9% | 58.3% | 210.29 | 18.02 | 212.29 |

### 全 prefix 网格

| branch | k | rows | safe success | task success | cause violation | 新动作 | actor calls |
|-|-|-|-|-|-|-|-|
| CACHED_MATCHED | 2 | 48 | 37.5% | 45.8% | 35.4% | 161.04 | 18.02 |
| CACHED_MATCHED | 3 | 48 | 4.2% | 6.2% | 68.8% | 251.40 | 17.98 |
| CACHED_MATCHED | 4 | 48 | 6.2% | 8.3% | 68.8% | 245.25 | 17.96 |
| CACHED_MATCHED | 5 | 48 | 10.4% | 10.4% | 58.3% | 238.85 | 17.92 |
| CACHED_MATCHED | 6 | 48 | 10.4% | 10.4% | 62.5% | 237.17 | 17.83 |
| CACHED_MATCHED | 7 | 48 | 10.4% | 12.5% | 56.2% | 231.33 | 17.79 |
| CACHED_MATCHED | 8 | 48 | 14.6% | 14.6% | 58.3% | 223.65 | 17.73 |
| CACHED_MATCHED | 9 | 48 | 14.6% | 14.6% | 58.3% | 222.94 | 17.71 |
| CACHED_MATCHED | 10 | 48 | 8.3% | 10.4% | 64.6% | 233.46 | 17.65 |
| CACHED_MATCHED | 11 | 48 | 12.5% | 16.7% | 66.7% | 218.65 | 17.65 |
| CACHED_MATCHED | 12 | 48 | 14.6% | 16.7% | 60.4% | 216.58 | 17.65 |
| CACHED_MATCHED | 13 | 48 | 20.8% | 20.8% | 50.0% | 206.44 | 17.60 |
| CACHED_MATCHED | 14 | 48 | 16.7% | 16.7% | 56.2% | 212.83 | 17.52 |
| CACHED_MATCHED | 15 | 48 | 20.8% | 22.9% | 52.1% | 198.92 | 17.52 |
| CACHED_MATCHED | 16 | 48 | 10.4% | 10.4% | 62.5% | 226.85 | 17.12 |
| CACHED_NOQUERY | 2 | 48 | 22.9% | 29.2% | 50.0% | 195.90 | 17.02 |
| CACHED_NOQUERY | 3 | 48 | 4.2% | 6.2% | 70.8% | 251.50 | 16.98 |
| CACHED_NOQUERY | 4 | 48 | 10.4% | 12.5% | 64.6% | 236.21 | 16.96 |
| CACHED_NOQUERY | 5 | 48 | 6.2% | 6.2% | 68.8% | 248.67 | 16.92 |
| CACHED_NOQUERY | 6 | 48 | 10.4% | 10.4% | 62.5% | 239.96 | 16.83 |
| CACHED_NOQUERY | 7 | 48 | 10.4% | 12.5% | 62.5% | 231.23 | 16.79 |
| CACHED_NOQUERY | 8 | 48 | 12.5% | 16.7% | 60.4% | 220.60 | 16.73 |
| CACHED_NOQUERY | 9 | 48 | 14.6% | 16.7% | 58.3% | 218.27 | 16.71 |
| CACHED_NOQUERY | 10 | 48 | 8.3% | 10.4% | 68.8% | 235.40 | 16.65 |
| CACHED_NOQUERY | 11 | 48 | 10.4% | 14.6% | 66.7% | 227.10 | 16.65 |
| CACHED_NOQUERY | 12 | 48 | 14.6% | 16.7% | 60.4% | 216.67 | 16.65 |
| CACHED_NOQUERY | 13 | 48 | 18.8% | 18.8% | 56.2% | 208.60 | 16.60 |
| CACHED_NOQUERY | 14 | 48 | 16.7% | 16.7% | 58.3% | 213.65 | 16.52 |
| CACHED_NOQUERY | 15 | 48 | 10.4% | 12.5% | 64.6% | 223.60 | 16.52 |
| CACHED_NOQUERY | 16 | 48 | 8.3% | 10.4% | 66.7% | 230.52 | 16.12 |
| FRESH_MATCHED | 2 | 48 | 16.7% | 22.9% | 58.3% | 210.29 | 18.02 |
| FRESH_MATCHED | 3 | 48 | 14.6% | 16.7% | 60.4% | 228.17 | 17.98 |
| FRESH_MATCHED | 4 | 48 | 14.6% | 16.7% | 58.3% | 225.54 | 17.96 |
| FRESH_MATCHED | 5 | 48 | 8.3% | 10.4% | 66.7% | 242.38 | 17.92 |
| FRESH_MATCHED | 6 | 48 | 10.4% | 12.5% | 66.7% | 237.33 | 17.83 |
| FRESH_MATCHED | 7 | 48 | 8.3% | 8.3% | 62.5% | 246.48 | 17.79 |
| FRESH_MATCHED | 8 | 48 | 12.5% | 14.6% | 62.5% | 230.56 | 17.73 |
| FRESH_MATCHED | 9 | 48 | 14.6% | 16.7% | 60.4% | 226.04 | 17.71 |
| FRESH_MATCHED | 10 | 48 | 10.4% | 10.4% | 60.4% | 244.06 | 17.65 |
| FRESH_MATCHED | 11 | 48 | 12.5% | 12.5% | 58.3% | 236.50 | 17.65 |
| FRESH_MATCHED | 12 | 48 | 14.6% | 14.6% | 56.2% | 230.56 | 17.65 |
| FRESH_MATCHED | 13 | 48 | 12.5% | 14.6% | 60.4% | 231.94 | 17.60 |
| FRESH_MATCHED | 14 | 48 | 10.4% | 12.5% | 62.5% | 236.00 | 17.52 |
| FRESH_MATCHED | 15 | 48 | 10.4% | 10.4% | 64.6% | 242.19 | 17.52 |
| FRESH_MATCHED | 16 | 48 | 8.3% | 10.4% | 64.6% | 241.42 | 17.12 |
| HOLD_PREFIX_MATCHED | 2 | 48 | 20.8% | 25.0% | 56.2% | 205.40 | 18.02 |
| HOLD_PREFIX_MATCHED | 3 | 48 | 10.4% | 10.4% | 64.6% | 243.81 | 17.98 |
| HOLD_PREFIX_MATCHED | 4 | 48 | 8.3% | 12.5% | 64.6% | 237.58 | 17.96 |
| HOLD_PREFIX_MATCHED | 5 | 48 | 2.1% | 6.2% | 70.8% | 254.00 | 17.92 |
| HOLD_PREFIX_MATCHED | 6 | 48 | 2.1% | 6.2% | 68.8% | 256.25 | 17.83 |
| HOLD_PREFIX_MATCHED | 7 | 48 | 6.2% | 12.5% | 66.7% | 240.73 | 17.79 |
| HOLD_PREFIX_MATCHED | 8 | 48 | 8.3% | 10.4% | 64.6% | 243.83 | 17.73 |
| HOLD_PREFIX_MATCHED | 9 | 48 | 10.4% | 12.5% | 62.5% | 239.58 | 17.71 |
| HOLD_PREFIX_MATCHED | 10 | 48 | 6.2% | 10.4% | 66.7% | 245.92 | 17.65 |
| HOLD_PREFIX_MATCHED | 11 | 48 | 10.4% | 12.5% | 62.5% | 239.85 | 17.65 |
| HOLD_PREFIX_MATCHED | 12 | 48 | 10.4% | 10.4% | 60.4% | 243.38 | 17.65 |
| HOLD_PREFIX_MATCHED | 13 | 48 | 10.4% | 12.5% | 60.4% | 239.81 | 17.60 |
| HOLD_PREFIX_MATCHED | 14 | 48 | 16.7% | 20.8% | 56.2% | 224.88 | 17.52 |
| HOLD_PREFIX_MATCHED | 15 | 48 | 10.4% | 12.5% | 62.5% | 239.58 | 17.52 |
| HOLD_PREFIX_MATCHED | 16 | 48 | 10.4% | 12.5% | 64.6% | 240.17 | 17.12 |

### 恢复 operator

| operator | rows | safe success | task success | cause violation | 新动作 | actor calls |
|-|-|-|-|-|-|-|
| fresh_h16 | 2160 | 0 | n/a | 10.0% | 11.2% | 63.1% |
| fresh_h4 | 2160 | 0 | n/a | 7.0% | 8.1% | 62.7% |
| hold_one_step_then_fresh_h16 | 2160 | 0 | n/a | 12.7% | 15.5% | 60.3% |
| rollback_one_step_then_fresh_h16 | 2160 | 0 | n/a | 7.2% | 8.9% | 64.2% |

### 可部署 baseline（由 calibration 冻结）

| baseline | rows | events | event coverage | safe success | 新动作 | actor calls | 平均 k |
|-|-|-|-|-|-|-|-|
| action_disagreement | 144 | 6.9% | 233.82 | 63.54 | 15.73 |  |  |
| fixed_delay_1 | 144 | 5.6% | 248.10 | 66.94 | 3.00 |  |  |
| fixed_delay_2 | 144 | 9.0% | 239.77 | 66.46 | 4.00 |  |  |
| fixed_delay_4 | 144 | 10.4% | 236.66 | 66.02 | 6.00 |  |  |
| fixed_delay_8 | 144 | 11.1% | 230.58 | 65.02 | 10.00 |  |  |
| immediate_fresh_h16 | 144 | 16.0% | 219.97 | 67.02 | 2.00 |  |  |
| k_last_observed_safe | 72 | 4.2% | 237.25 | 61.92 | 15.96 |  |  |
| k_last_recoverable | 54 | 37.0% | 180.15 | 67.94 | 13.44 |  |  |
| velocity_phase | 144 | 11.1% | 231.17 | 64.85 | 10.90 |  |  |

## Track A：compute-matched cached prefix

Track A boundary rows 总计=39，其中 evaluation rows=18。在被 atlas 保留的子集上，cached−fresh safe-success 描述性差为 `0.1833`，10,000 次 `(task, init_state_id)` cluster bootstrap 95% CI 为 `[0.0500, 0.3333]`；逐 row 新动作 reduction fraction 的均值为 `-694.3%`。两任务 recoverable-window 描述性中位数为 `{'put_the_bowl_on_the_stove': 14.0, 'put_the_cream_cheese_in_the_bowl': 13.0}`。由于 stove 的 33 个 replay-invalid 事件被边界构造排除，这不是完整样本的有效效应估计。

所有 planned prefix row 的账面 compute-matched=`True`、budget-matched=`True`；这只验证预算字段，不能修复运行态污染。`CACHED_NOQUERY` 与 matched cached 的最终安全结果一致率为 `91.7%`，平均少 `1.00` 次 actor call；由于它们连 pre-tail `S_obs` 都可能不同，本轮不能称为 deployment evidence。

## Phase E：operator-relative recoverability

atlas rows=1440，event boundaries=96，replay-invalid events=33。`k_last_recoverable` 只对冻结恢复算子族 U 定义；`k_irrev_U` 是该算子族下的 persistent crossing，原始非单调恢复 pattern 完整保留。由于 prefix safety 本身受 reset/order 污染，这些边界只能描述该次运行的 frozen operator rows；它不是物理不可逆点。

## Track B：cross-fitted replanability

LOAO cross-fit rows 总计=117，其中 evaluation rows=54；calibration-only 冻结最强 baseline 为 `k_last_recoverable`。但 calibration 排名并非 common-support：该 boundary baseline 只覆盖 `21/48` 个事件，固定延迟类各覆盖 48/48；evaluation 也只有 `18/48` 个事件、`54` 个 actor rows。held-out safe-success 描述性差为 `-0.0944`，95% CI=`[-0.2333, 0.0389]`；逐 row 新动作 reduction fraction 均值为 `-125.4%`。

选择分布 `{'2': 4, '4': 7, '6': 2, '7': 3, '8': 3, '9': 2, '10': 3, '11': 3, '12': 6, '13': 2, '14': 1, '15': 7, '16': 11}`；baseline 分布 `{'4': 6, '9': 3, '11': 3, '12': 3, '13': 3, '15': 9, '16': 27}`；与 d 不同 `92.6%`，与 `k_last_recoverable` 不同 `72.2%`，内部 prefix `72.2%`。held-out actor 的 outcome 在选 k 后才读取。

## 10,000 次分层 cluster bootstrap

独立 cluster 始终是 `(task, init_state_id)`；actor seed 是 cluster 内重复测量，prefix row 不是独立样本。任务、严重度、baseline、阈值均未用 evaluation outcome 重新选择。bootstrap 实现正确，但输入已受 replay contract 和有效子集选择限制，因此区间只作描述。

### Track A

| 指标 | 分层 | rows | clusters | 估计 | 95% CI |
|-|-|-|-|-|-|
| 安全成功差 | 全部 cluster | 18 | 10 | 0.1833 | [0.0500, 0.3333] |
| 安全成功差 | 任务宏平均 | 18 | 10 | 0.2083 | [0.0312, 0.3854] |
| 安全成功差 | 任务: put_the_bowl_on_the_stove | 15 | 8 | 0.1667 | [0.0417, 0.3333] |
| 安全成功差 | 任务: put_the_cream_cheese_in_the_bowl | 3 | 2 | 0.2500 | [0.0000, 0.5000] |
| 安全成功差 | 任务/严重度: put_the_bowl_on_the_stove/future_06_lateral_040mm | 8 | 7 | 0.2857 | [0.0000, 0.5714] |
| 安全成功差 | 任务/严重度: put_the_bowl_on_the_stove/future_14_lateral_020mm | 7 | 6 | 0.0833 | [-0.4167, 0.5000] |
| 安全成功差 | 任务/严重度: put_the_cream_cheese_in_the_bowl/shift_040mm | 2 | 2 | 0.0000 | [0.0000, 0.0000] |
| 安全成功差 | 任务/严重度: put_the_cream_cheese_in_the_bowl/shift_060mm | 1 | 1 | 1.0000 | [1.0000, 1.0000] |
| 安全成功差 | actor seed: 17 | 7 | 7 | 0.4286 | [0.1429, 0.8571] |
| 安全成功差 | actor seed: 29 | 5 | 5 | 0.4000 | [0.0000, 0.8000] |
| 安全成功差 | actor seed: 7 | 6 | 6 | -0.1667 | [-0.5000, 0.0000] |
| 新非名义动作差 | 全部 cluster | 18 | 10 | 73.9167 | [28.6492, 121.4000] |
| 新非名义动作差 | 任务宏平均 | 18 | 10 | 111.3542 | [72.4367, 150.3677] |
| 新非名义动作差 | 任务: put_the_bowl_on_the_stove | 15 | 8 | 48.9583 | [12.8943, 89.8750] |
| 新非名义动作差 | 任务: put_the_cream_cheese_in_the_bowl | 3 | 2 | 173.7500 | [122.5000, 225.0000] |
| 新非名义动作差 | 任务/严重度: put_the_bowl_on_the_stove/future_06_lateral_040mm | 8 | 7 | 78.8571 | [8.2857, 152.2857] |
| 新非名义动作差 | 任务/严重度: put_the_bowl_on_the_stove/future_14_lateral_020mm | 7 | 6 | 27.5833 | [-105.9167, 157.0833] |
| 新非名义动作差 | 任务/严重度: put_the_cream_cheese_in_the_bowl/shift_040mm | 2 | 2 | 119.5000 | [14.0000, 225.0000] |
| 新非名义动作差 | 任务/严重度: put_the_cream_cheese_in_the_bowl/shift_060mm | 1 | 1 | 231.0000 | [231.0000, 231.0000] |
| 新非名义动作差 | actor seed: 17 | 7 | 7 | 116.4286 | [43.5714, 218.4286] |
| 新非名义动作差 | actor seed: 29 | 5 | 5 | 147.8000 | [54.2000, 241.4000] |
| 新非名义动作差 | actor seed: 7 | 6 | 6 | -37.0000 | [-128.3333, 10.8333] |

### Track B

| 指标 | 分层 | rows | clusters | 估计 | 95% CI |
|-|-|-|-|-|-|
| 安全成功差 | 全部 cluster | 54 | 10 | -0.0944 | [-0.2333, 0.0333] |
| 安全成功差 | 任务宏平均 | 54 | 10 | -0.2153 | [-0.2917, -0.1389] |
| 安全成功差 | 任务: put_the_bowl_on_the_stove | 45 | 8 | -0.0139 | [-0.1250, 0.0972] |
| 安全成功差 | 任务: put_the_cream_cheese_in_the_bowl | 9 | 2 | -0.4167 | [-0.5000, -0.3333] |
| 安全成功差 | 任务/严重度: put_the_bowl_on_the_stove/future_06_lateral_040mm | 24 | 7 | 0.1190 | [-0.1190, 0.3571] |
| 安全成功差 | 任务/严重度: put_the_bowl_on_the_stove/future_14_lateral_020mm | 21 | 6 | -0.0556 | [-0.1667, 0.0000] |
| 安全成功差 | 任务/严重度: put_the_cream_cheese_in_the_bowl/shift_040mm | 6 | 2 | -0.3333 | [-0.3333, -0.3333] |
| 安全成功差 | 任务/严重度: put_the_cream_cheese_in_the_bowl/shift_060mm | 3 | 1 | -0.6667 | [-0.6667, -0.6667] |
| 安全成功差 | actor seed: 17 | 18 | 10 | -0.1667 | [-0.4000, 0.0333] |
| 安全成功差 | actor seed: 29 | 18 | 10 | -0.1000 | [-0.3500, 0.1000] |
| 安全成功差 | actor seed: 7 | 18 | 10 | -0.0167 | [-0.2000, 0.1667] |
| 新非名义动作差 | 全部 cluster | 54 | 10 | -24.6722 | [-58.7008, 6.5446] |
| 新非名义动作差 | 任务宏平均 | 54 | 10 | -54.2951 | [-71.6146, -37.4271] |
| 新非名义动作差 | 任务: put_the_bowl_on_the_stove | 45 | 8 | -4.9236 | [-30.5833, 19.6262] |
| 新非名义动作差 | 任务: put_the_cream_cheese_in_the_bowl | 9 | 2 | -103.6667 | [-121.3333, -86.0000] |
| 新非名义动作差 | 任务/严重度: put_the_bowl_on_the_stove/future_06_lateral_040mm | 24 | 7 | 25.6667 | [-26.0571, 84.6667] |
| 新非名义动作差 | 任务/严重度: put_the_bowl_on_the_stove/future_14_lateral_020mm | 21 | 6 | -14.7778 | [-43.2236, 0.2222] |
| 新非名义动作差 | 任务/严重度: put_the_cream_cheese_in_the_bowl/shift_040mm | 6 | 2 | -90.0000 | [-94.0000, -86.0000] |
| 新非名义动作差 | 任务/严重度: put_the_cream_cheese_in_the_bowl/shift_060mm | 3 | 1 | -148.6667 | [-148.6667, -148.6667] |
| 新非名义动作差 | actor seed: 17 | 18 | 10 | -43.0167 | [-103.5500, 8.9333] |
| 新非名义动作差 | actor seed: 29 | 18 | 10 | -26.1500 | [-85.6500, 22.4500] |
| 新非名义动作差 | actor seed: 7 | 18 | 10 | -4.8500 | [-48.8500, 38.1000] |

## 为什么会提升或降低：冻结代码的反向解释

evaluation 中共有 720 个 planned cached/fresh prefix pair：原始 cached outcome 改善安全结果 52 个，恶化 36 个，不变 632 个。它们是结果分桶，不是 replay-valid 因果 pair。

| 结果组 | n | safe Δ | cause Δ | action disagreement | cached−fresh progress | fresh−cached 新动作 | 完成步数 Δ | 接触 Δ |
|-|-|-|-|-|-|-|-|-|
| cached improves | 52 | 1.0000 | -1.0000 | 0.1434 | 0.0241 | 244.0962 | -237.3462 | -4983.1346 |
| cached worsens | 36 | -1.0000 | 1.0000 | 0.1659 | -0.0169 | -234.6667 | 240.1944 | 5140.7778 |
| same | 632 | 0.0000 | -0.0079 | 0.2369 | -0.0001 | 7.2658 | -0.1614 | -85.8370 |

这项反解先做排除：同一 cached prefix 的 pre-operator `S_obs` 会随分支顺序改变，因此无法把改善归因于保留名义进度，也无法把恶化归因于 stale action。表中 progress、cause、displacement、路径和接触只说明污染后的数值如何分组；它们不能区分动作机制与 reset/order 状态。它不是新 idea，也不是 learned mediator。

必须特别警惕结构性指标：`new_non_nominal_actions` 把保留的 cached `A[d:k]` 计为旧动作，而把 fresh prefix/tail 计为新动作，所以 k 越大时该指标天然更容易下降。cross-fit 的字典序在安全结果并列后依次偏好更少新动作、更少 actor call、最后更小 k，因此安全并列时可能被这个定义推向较晚 prefix。新动作减少若没有独立安全/任务提升，不能当成增益机理。

## 判定向量

| 字段 | 值 |
|-|-|
| accepted | False |
| all_experiments_continued_after_failed_gates | True |
| contract_repair | PASS |
| frozen_strongest_baseline | k_last_recoverable |
| local_repair | RETIRED_NO_SIGNAL |
| novelty | N2_ORACLE_PROTOCOL_BOUNDARY_ONLY |
| operator_router | RETIRED_NO_SIGNAL |
| overall | BLOCKED_BY_REPLAY_CONTRACT |
| replay_contract | BLOCKED |
| schema_version | 1 |
| second_failure_family | BLOCKED |
| stage1b_universal_hypothesis | KILLED_IMMUTABLE |
| stage2b_status | BLOCKED_UPSTREAM |
| track_a_operator_relative_prefix_reuse | INCONCLUSIVE |
| track_a_raw_signal | False |
| track_b_crossfit_replanability | INCONCLUSIVE |
| track_b_raw_signal | False |

### Upstream gates

| gate | 结果 |
|-|-|
| contract_repair | PASS |
| formal_matrix | PASS |
| minimum_data | PASS |
| replay_contract | FAIL |
| second_failure_family | FAIL |

### Track A 原始 criteria

| criterion | 结果 |
|-|-|
| all_budget_matched | PASS |
| all_compute_matched | PASS |
| both_tasks_median_window_ge_2 | PASS |
| direction_all_severities | FAIL |
| direction_both_tasks | FAIL |
| direction_two_of_three_generator_seeds | PASS |
| new_actions_reduced_ge_15pct | FAIL |
| safe_ci_lower_ge_minus_0_03 | PASS |

### Track B 原始 criteria

| criterion | 结果 |
|-|-|
| action_and_call_budgets_equal | FAIL |
| claimed_gain_ci_lower_gt_0 | FAIL |
| interior_selected_ge_20pct | PASS |
| positive_both_tasks | FAIL |
| positive_two_of_three_heldout_seeds | FAIL |
| primary_gain_threshold | FAIL |
| selected_differs_from_d_ge_20pct | PASS |
| selected_differs_from_last_recoverable_ge_20pct | PASS |

## 证据边界与停止点

- local repair 与 operator router 保持 retired；本轮没有调参、训练或给予正向标签。
- learned/deployable evidence 仍为 **NONE**。replay contract 失败后，`CACHED_NOQUERY` 也只能是受污染的离线/模拟诊断。
- 33/96 个事件的运行态分支隔离失败；即使某些 raw CI 看似正向，也不得越过该 gate。
- 这不是 VLA、world model、π0.5 或真实机器人实验；没有闭环部署或物理不可逆性证明。
- Stage-1b universal kill 不可逆转；Stage-2B blocked/inconclusive 不被覆盖。
- 最终 `accepted=false`，novelty 不高于 N2。按 Stage-2C 停止点到此结束，不自动进入 learned replanability。

## 可复现产物

代码、原始 attempts、actor events、replay traces、全部 matched/recovery/cross-fit rows、10,000-draw 统计、`replay_contract_diagnostic.*`、机理审计、测试输出和 SHA256 清单均保存在本仓库的 `experiments/r16_p14_stage2c/` 与 `artifacts/stage2c/`。PAI 原始报告另存为 `REPORT_PAI_RAW.md`；PAI v1–v8 和独立 replay probe 的每次失败/停止/成功记录均保留，未隐藏基础设施失败。
