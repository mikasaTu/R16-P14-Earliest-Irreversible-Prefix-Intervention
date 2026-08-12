# R16-P14 Stage-1 Oracle Prefix Feasibility Audit：LIBERO 初步实验报告

## 结论摘要

本轮实验的预注册决策为：

`REVISE_TASKS_OR_PERTURBATIONS`

在 3 个 LIBERO-GOAL 任务上，只有“将碗放到盘子上”表现出明确的非零可干预窗口：median intervention window 为 4 个动作，median safe-prefix retention 为 50%，oracle intervention 将 violation 从 17 降到 6（相对下降 64.7%）。另外两个任务的 median window 和 safe-prefix retention 都为 0。

汇总后，oracle intervention 将 unsafe outcome 从 72 降到 56（相对下降 22.2%），但 suffix-only intervention 相对 same-trigger full replan 的 median rework reduction 仅 4.1%。7 个预注册 gate 中只有“至少两类 failure cause 的最佳 operator 不同”通过。因此，本轮结果说明 prefix intervention 机制在特定任务上有信号，但证据不足以进入 learned prefix-risk model 训练，更不能视为最终算法有效性的证明。

## 1. 要验证的 idea

原始假设关注 action chunk `A = [a1, ..., aH]` 执行期间的两个边界：

- `k_irrev_physical`：从某个 prefix 后，即使允许 privileged scripted recovery，也无法在固定预算内避免指定 failure cause 并完成任务的最早位置。
- `k_irrev_policy`：从某个 prefix 后，部署时允许的 intervention operators 与当前 BC policy 已无法在固定预算内恢复的最早位置。

二者用于构造 intervention window，并回答四个 Stage-1 问题：是否存在非零可干预窗口、已安全执行的 prefix 是否值得保留、只取消 suffix 是否比 whole-chunk replan 少返工，以及不同 failure cause 是否偏好不同 intervention operator。

本阶段只做 oracle feasibility audit。没有训练 risk head，没有使用 VLA、π0.5 或 world model，也没有提出最终性能提升主张。

## 2. 为什么改用 LIBERO

按实验要求将原始 ManiSkill3 方案迁移到 LIBERO-GOAL，并选择三种机制互补的开发任务：

| LIBERO 任务 | 对应 failure cause | 选择理由 |
| --- | --- | --- |
| `open_the_middle_drawer_of_the_cabinet` | mechanism path obstacle | 有受约束机构运动，适合测试路径阻挡后的 stop/retract/replan |
| `put_the_bowl_on_the_plate` | target shift or premature release | 包含运输、对齐与释放，适合测试保留早期安全搬运动作 |
| `put_the_wine_bottle_on_the_rack` | grasp slip or contact misalignment | 接触与姿态约束明显，适合测试稳定抓取和局部修复 |

`open_the_top_drawer_and_put_the_bowl_inside` 被预注册为 held-out compositional task，本轮未用它调参或给出结果。

## 3. 实验协议

### 3.1 Policy 与训练

- Policy：只使用 `robot0_proprio-state` 和 `object-state` 的小型 chunked BC MLP。
- Action chunk length：16。
- Policy seeds：7、17、29。
- 每个 task/seed 使用 50 条 demonstration，AdamW，learning rate `3e-4`，batch size 256。
- 每个 task/seed 训练 8,000 optimizer steps，共 9 个模型。
- 每 2,000 steps 保存一次 full-state checkpoint；发布包保留 9 个最终 checkpoint，包含 model、optimizer、scheduler、RNG、normalization 和 global step。

### 3.2 Clean baseline

- 固定 execution horizons：1、4、8、16。
- 每个 task/seed/horizon 运行 10 个 episode。
- 总量：3 tasks × 3 seeds × 4 horizons × 10 episodes = 360 rollouts。
- 这是 bounded pilot；原始 final gate 要求每个 seed/horizon 50 个 episode。

### 3.3 Perturbation 与 oracle branch replay

- 扰动只改变环境，不修改 action token。
- 预注册 insertion prefixes：2、6、10；prefix stride 为 2。
- 每个任务最多 30 个 oracle candidates，本轮实际共 90 个。
- 每个 branch 的统一预算为 96 simulator steps。
- 比较的 policy-side operators 包括 nominal continue、trim and replan、hold and replan、bounded rollback and replan，以及 cause-specific local repair。
- Oracle audit 使用 policy seed 7；clean baseline 使用全部 3 个 policy seeds。

### 3.4 可复现性保护

- Instrumentation parity：同一 seed 下，开关 instrumentation 的 action、chunk、state/contact trace、success 与 policy-call count 必须一致；3 个任务均通过。
- Snapshot replay：从相同 simulator state 重放 suffix，检查最终 state、outcome 与 contact trace。
- Source、LIBERO config、3 个 demonstration HDF5、PAI payload 和关键结果文件均用 SHA-256 冻结。

## 4. 实验结果

### 4.1 Clean baseline

每个表格单元包含 30 个 episode（3 seeds × 10）。

| Task | h=1 | h=4 | h=8 | h=16 | 合计 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Open middle drawer | 1/30 (3.3%) | 3/30 (10.0%) | 3/30 (10.0%) | 3/30 (10.0%) | 10/120 (8.3%) |
| Bowl on plate | 2/30 (6.7%) | 2/30 (6.7%) | 2/30 (6.7%) | 0/30 (0.0%) | 6/120 (5.0%) |
| Wine bottle on rack | 1/30 (3.3%) | 5/30 (16.7%) | 5/30 (16.7%) | 0/30 (0.0%) | 11/120 (9.2%) |

三项 baseline 都明显低于原方案偏好的 30%–90% success 区间，这是本轮不能直接进入 learned-risk 阶段的第一项重要限制。不同 horizon 仍产生了可区分结果，但这些差异不能被解释为统计显著的 horizon 优势。

### 4.2 Oracle prefix audit

| Task | Candidates | Usable unsafe / near-unsafe | Median window | Safe-prefix retention | Unsafe：无干预 → oracle | 相对下降 | 相对 full replan 的 rework reduction | Replay | Best operator |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Open middle drawer | 30 | 27 | 0 | 0.0% | 27 → 26 | 3.7% | 0.0% | 100.0% | trim and replan |
| Bowl on plate | 30 | 17 | 4 | 50.0% | 17 → 6 | 64.7% | 8.2% | 86.7% | bounded rollback and replan |
| Wine bottle on rack | 30 | 28 | 0 | 0.0% | 28 → 24 | 14.3% | N/A（无有效 paired comparison） | 100.0% | bounded rollback and replan |

关键观察：

1. 碗任务提供了本轮最强的正信号。它同时具有非零窗口、50% prefix retention 和显著的 violation reduction，说明“已经安全完成的动作不应全部丢弃”在该机制上值得继续研究。
2. 抽屉和酒瓶任务的大多数有效边界落在 chunk 开端，现有 task/perturbation/policy 组合没有给 oracle 留出可利用的 prefix window。
3. 返工降低未达到目标。三个任务的汇总 median relative rework reduction 为 4.1%，远低于 20% gate。
4. 碗任务 replay determinism 只有 86.7%，低于 99% gate；因此其正信号需要在更可靠的 replay 条件下复核。

### 4.3 汇总指标

| 指标 | 结果 |
| --- | ---: |
| Clean rollouts | 360 |
| Oracle candidates | 90 |
| No-intervention unsafe outcomes | 72 |
| Oracle unsafe outcomes | 56 |
| Relative unsafe-outcome reduction | 22.2% |
| Median relative rework reduction | 4.1% |
| Median safe-prefix retention | 0.0% |
| Tasks with median window ≥ 2 | 1 / 3 |
| Minimum replay determinism | 86.7% |

### 4.4 预注册 gates

| Gate | 阈值 | 结果 | 状态 |
| --- | --- | --- | --- |
| Minimum usable unsafe chunks | 每个任务 ≥ 30 | 27 / 17 / 28 | FAIL |
| Nonzero window across tasks | 至少 2 个任务 median window ≥ 2 | 1 个任务 | FAIL |
| Unsafe reduction | ≥ 30% | 22.2% | FAIL |
| Rework reduction vs same-trigger full replan | ≥ 20% | median 4.1% | FAIL |
| Safe-prefix retention | median ≥ 30% | 0.0% | FAIL |
| Cause-specific operator diversity | 至少两类 cause 的 winner 不同 | trim vs bounded rollback | PASS |
| Replay determinism | ≥ 99% | minimum 86.7% | FAIL |

总计 1 PASS / 6 FAIL。

## 5. 决策与下一步

本轮不支持 `PROCEED_TO_LEARNED_RISK`，但也不足以支持 `KILL_PREFIX_HYPOTHESIS`。原因是碗任务出现了机制上有意义的正信号，而另外两个任务同时受到 baseline 过弱和 intervention window 为零的混杂影响。因此选择 `REVISE_TASKS_OR_PERTURBATIONS`。

建议的 Stage-1b 顺序为：

1. 先把 clean baseline 提升到约 30%–90%，或至少稳定完成主要前缀后再失败；在此之前不训练 learned risk head。
2. 优先修复碗任务的 snapshot/replay 稳定性并原样复跑，确认 4-action window 与 50% retention 不是 replay artifact。
3. 重新标定抽屉和酒瓶的 perturbation timing/severity，使 failure 发生在 chunk 中段而不是 `k≈0`；保持扰动为 environment-only。
4. 将每个 task/seed/horizon 的 clean episode 从 10 补到预注册的 50，并扩充 oracle candidate pool。
5. 在 revised task set 上重新要求全部 gate，通过后才考虑 learned prefix-risk model。

## 6. 证据边界

- 这是初步机制资格检查，不是最终算法评测。
- Clean pilot 只有最终预注册 episode 数的 1/5。
- Oracle 使用 privileged branch replay；线上系统不能直接获得这些标签。
- Physical recoverability 是 scripted proxy，不是对连续动力学可恢复性的穷举证明。
- Policy 是 state-observation BC，不是目标 VLA；结论不能直接外推到 RGB/proprio VLA。
- 没有在 held-out compositional task 上报告结果。
- 本报告中的比例是描述性结果，未做置信区间或显著性检验。

## 7. 执行与产物来源

正式 PAI run 为 `r16p14-libero-stage1-20260812-144626`，Job ID 为 `dlc1l9akne34qq7k`。任务在 2×NVIDIA A800、12 CPU、200 GiB memory/shared-memory 合约下运行，2026-08-12 14:47:29 UTC 开始，15:41:47 UTC 成功结束，总时长 3,308 秒。AIMaster、elastic training 与自动 fault tolerance 均关闭，观测到 0 次平台重启。

正式实验冻结在 source commit `a1b61194a8382f5b1a247b9cd9b140645ff2aeb8`、tree `53001c43fbbb165c0a1f2c71f9cbd4c81b9d0ced`。详细文件映射和验证命令见 [REPRODUCIBILITY.md](REPRODUCIBILITY.md)，原始需求全文保存在 [ORIGINAL_STAGE1_BRIEF.txt](ORIGINAL_STAGE1_BRIEF.txt)。

飞书“实验报告”子文档：[R16-P14 Stage-1 Oracle Prefix Feasibility Audit — LIBERO 初步实验报告](https://icnbwz7kd1ui.feishu.cn/wiki/DOVIwBUrZi4RAskJW6CcJpOLnif)。
