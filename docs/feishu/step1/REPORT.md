<title>实验报告</title>

**结论：REVISE_TASKS_OR_PERTURBATIONS。** 本轮 LIBERO Stage-1 oracle-prefix bounded pilot 找到了局部正信号，但没有通过预注册的可行性门槛，因此不应进入 learned risk / VLA / world-model 阶段，也不应直接扩大到最终 50 episodes/seed/horizon。当前最合理的动作是先修复 replay、重新校准任务与扰动，再做一次同规模复验。

对应 idea：[R16-P14：Earliest-Irreversible-Prefix Intervention](https://icnbwz7kd1ui.feishu.cn/wiki/MQWAwpio6iIKh3k7oWQcP4qOnLb)。本报告只回答 Stage-1 的 oracle feasibility，不证明最终算法有效。

# 1. 验证问题与判定

核心问题是：对长 action chunk 注入可控扰动后，是否存在一个非平凡的最早不可逆前缀；如果保留安全前缀、只干预后缀，能否相对“不干预”和“同触发时刻全量重规划”同时降低 unsafe outcome 与 rework。

- **正信号：** Bowl-on-plate 的 median intervention window 为 4/16，median safe-prefix retention 为 50%，oracle unsafe reduction 为 64.7%。说明“安全前缀可保留、危险后缀可被干预”的现象在至少一个任务上存在。
- **负信号：** Drawer 与 Wine 的 median window 均为 0；三任务合并 unsafe reduction 只有 22.2%；median rework reduction 只有 4.1%；Bowl replay determinism 只有 86.7%。
- **判定：** 现象不是完全不存在，但当前任务/扰动/恢复算子组合不够稳定，不能支持进入 Stage-2。

# 2. Benchmark 与任务选择

将原规划中的 ManiSkill3 换为 LIBERO-GOAL，选取三个具有不同失效机制的任务：

1. **open_the_middle_drawer_of_the_cabinet**：机构约束与阻挡，主要测试 mechanism obstruction。
2. **put_the_bowl_on_the_plate**：目标位置变化与过早释放，主要测试 target shift / premature release。
3. **put_the_wine_bottle_on_the_rack**：抓取滑移与接触对齐，主要测试 grasp slip / contact alignment。

预留但未在本 bounded pilot 执行的 held-out 任务：open_the_top_drawer_and_put_the_bowl_inside。

# 3. 预注册实验契约

- Action chunk 长度 H=16；clean rollout 执行 horizon 为 1、4、8、16。
- Policy seeds：7、17、29；每个 task × seed × horizon 为 10 episodes，共 360 条 clean rollout。
- 每任务 30 个 oracle candidates，共 90 个；insertion prefixes 为 2、6、10；prefix stride=2；branch budget=96；oracle execution horizon=8。
- Oracle operators：nominal continue、hold-and-replan、trim-and-replan、bounded rollback-and-replan、cause-specific local repair；另有 privileged scripted recovery 作为 physical recoverability proxy。
- Policy：state-observation chunked BC MLP，hidden dim 512，batch 256，每 task/seed 8000 steps，checkpoint cadence 2000。
- 本轮是 bounded pilot：clean volume 为最终门槛规划的 10/50。结果足以作“不扩大”的决策，但不能当作最终统计结论。

# 4. 开发机 smoke 与复现性检查

- 5 个自动化测试全部通过：chunk boundary、partial checkpoint ignore、model shape、simulator replay、3-task aggregate/decision。
- 同 snapshot + suffix 三次 replay 最大状态误差约 3e-14；oracle smoke replay error 为 0。
- 官方 drawer demo 可成功回放；20-step 小训练 loss 从 0.3896 降至 0.3366。
- step 10 原子 checkpoint 后受控退出，再自动恢复至 step 20，通过 full-state resume 检查。
- instrumentation enabled/disabled 的 action、state、contact hash 在三任务 PAI rollout 前置检查中全部一致。

# 5. PAI 训练与推理契约

- **最终成功 run：** r16p14-libero-stage1-20260812-144626。
- **PAI Job ID：** dlc1l9akne34qq7k；终态 Succeeded；运行约 3308 秒。
- **资源：** 专用 exp-efficiency，ResourceId=quotakzri8a5wqcp，1 worker，2×A800，12 CPU，200 GiB RAM/shared memory。
- **故障契约：** AIMaster 关闭、平台自动重启=0、单一 Pod UID、无 launcher 重试。
- **运行时：** UID/GID 2254:2254；Torch 2.5.1+cu124；CUDA 12.4；robosuite 1.4；MuJoCo 3.6。
- **冻结源码：** commit a1b61194a8382f5b1a247b9cd9b140645ff2aeb8；tree 53001c43fbbb165c0a1f2c71f9cbd4c81b9d0ced；最终 PAI payload SHA-256 为 3c4c83dc5c02b5b9ef780530589e22c4b451a2ab446f043112b224c2c715990a。
- **Application run ID：** r16p14-libero-stage1-pilot-v1；九个 step-8000 checkpoint 均通过内容 SHA-256 与 .complete.json 校验。
- **W&B：**[r16-p14-libero-stage1 project](https://wandb.ai/chen_jian-cj-workspace/r16-p14-libero-stage1)。

# 6. 训练结果

PAI 上首个真实更新证据：Bowl seed 7，global step 1，loss=0.43323937，learning rate≈3.0e-4。

- **Drawer：** seed 7/17/29 的 step-8000 loss 分别为 0.0043585、0.0046133、0.0044846。
- **Bowl：** seed 7/17/29 分别为 0.0087654、0.0084766、0.0083378。
- **Wine：** seed 7/17/29 分别为 0.0099415、0.0105948、0.0092241。

训练 loss 收敛只证明 BC 拟合正常，不等价于 rollout 成功或 oracle feasibility。

# 7. Clean rollout 结果

每个单元为 30 episodes（3 policy seeds × 10 episodes）。

- **Drawer：** h=1 为 1/30（3.3%）；h=4 为 3/30（10.0%）；h=8 为 3/30（10.0%）；h=16 为 3/30（10.0%）。总计 10/120。
- **Bowl：** h=1 为 2/30（6.7%）；h=4 为 2/30（6.7%）；h=8 为 2/30（6.7%）；h=16 为 0/30（0%）。总计 6/120。
- **Wine：** h=1 为 1/30（3.3%）；h=4 为 5/30（16.7%）；h=8 为 5/30（16.7%）；h=16 为 0/30（0%）。总计 11/120。

所有 clean rollout 的 success 与 safe_success 计数一致。整体成功率偏低，说明当前小型 state-BC 是 Stage-1 的重要限制项；不能把 oracle 的改善幅度外推到更强 VLA。

# 8. Oracle prefix audit 结果

## 8.1 Drawer / mechanism obstruction

- 30 candidates；usable unsafe/near-unsafe=27。
- Median intervention window=0；median safe-prefix retention=0%。
- No-intervention violations 27 → oracle violations 26；relative reduction=3.7%。
- Paired rework comparisons=1；relative rework reduction=0%。
- Replay determinism=100%；主要胜出算子为 trim-and-replan（1 次）。

## 8.2 Bowl / target shift

- 30 candidates；usable=17。
- Median intervention window=4；median safe-prefix retention=50%。
- No-intervention violations 17 → oracle violations 6；relative reduction=64.7%。
- Paired rework comparisons=3；relative rework reduction=8.2%。
- Replay determinism=86.7%；算子胜出：bounded rollback 6、hold-and-replan 3、local repair 1、trim 1。

## 8.3 Wine / grasp slip

- 30 candidates；usable=28。
- Median intervention window=0；median safe-prefix retention=0%。
- No-intervention violations 28 → oracle violations 24；relative reduction=14.3%。
- Paired rework comparisons=0，因此 rework reduction 不可估计。
- Replay determinism=100%；算子胜出：bounded rollback 2、local repair 1、hold-and-replan 1。

## 8.4 跨任务聚合

- Unsafe outcomes：72 → 56，relative reduction=22.2%（门槛 30%）。
- Median relative rework reduction=4.1%（门槛 20%）。
- Median safe-prefix retention=0%（门槛 30%）。
- Median window ≥2 的任务只有 1/3（要求至少 2/3）。
- 最小 replay determinism=86.7%（门槛 99%）。

# 9. Gate-by-gate 判定

- **FAIL** — 每任务至少 30 usable：27 / 17 / 28。
- **FAIL** — 至少两个任务 median window ≥2：只有 Bowl。
- **FAIL** — relative unsafe reduction ≥30%：22.2%。
- **FAIL** — relative rework reduction ≥20%：4.1%。
- **FAIL** — median safe-prefix retention ≥30%：0%。
- **PASS** — 至少两个不同 cause/operator winner：trim-and-replan 与 bounded rollback-and-replan。
- **FAIL** — replay determinism ≥99%：最低 86.7%。

**总判定：1 PASS / 6 FAIL，REVISE_TASKS_OR_PERTURBATIONS。**

# 10. 解释与下一步

1. **先修 replay：** 对 Bowl 的 snapshot 恢复补齐 simulator/control state，记录首次 hash divergence；在 replay ≥99% 前不重新解释其 64.7% 改善。
2. **重做 candidate admission：** 排除扰动后已经 terminal success 的平凡 candidate；要求每任务稳定获得 ≥30 个真正 unsafe/near-unsafe chunks。
3. **校准 anchor 与强度：** anchor 放到首次关键接触/释放之前；做有方向对称的 severity sweep，避免 intensity 与方向混杂。
4. **提高 baseline competence：** 先让小策略在所选任务达到更可解释的 clean success，再评价 prefix intervention；暂不扩大到大模型训练。
5. **任务替换候选：** 可用 held-out 的 open_the_top_drawer_and_put_the_bowl_inside，或选择 demo replay/BC 成功率更高且关键接触时刻清晰的 LIBERO task，替换当前 window=0 的弱任务。
6. **停止条件：** 只有同规模复验通过 replay、usable、window、unsafe reduction 与 rework gates，才扩到最终 50 episodes/seed/horizon 或训练 learned risk head。

# 11. 限制

- 本轮 clean pilot 为每 seed/horizon 10 episodes，不是最终规划的 50。
- Oracle audit 只使用 policy seed 7；clean baseline 覆盖三个 seeds。
- Physical recoverability 是 privileged scripted-recovery proxy，不是穷尽动力学的证明。
- 本轮策略是 state-observation chunked BC；没有测试 learned risk、VLA 或 world model。
- 低 clean success 会限制 oracle 结果的外部有效性。

# 12. 审计链与原始产物

- Artifact root：/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r16_p14_libero_stage1/pai/r16p14-libero-stage1-20260812-144626
- Checkpoint lineage：/mnt/cpfs/zbl-cpfs-new/CKPT/leon/torch/r16_p14_libero_stage1/r16p14-libero-stage1-pilot-v1
- experiment_summary.json SHA-256：0b50014b608dfd7c40a6ef73251c59238c19890b38b3182a41088b65bbc08fb9
- EVALUATION_COMPLETE.json SHA-256：819d04e6f0a3c02d59a23ff513f2c70c7db5330ffa7a0b5f5fb586f6b92b8248
- TRAINING_COMPLETE.json SHA-256：b745bf49564970c8b3fdd85bf53774346973c39b4cfaea5823e1053d71e105d6

提交恢复审计：两个本地 preflight 在 CreateJob 前 fail-closed，无 Job ID；三个被替代 Job 均在 step 1 前终止，原因依次为 CUDA runtime/EGL 契约、降权后 cwd 权限、源码路径预检断言。它们均为 Failed 终态、无活动 Pod、没有完整 checkpoint 污染；最终成功 Job 使用同一 application lineage 且通过完整性校验。
