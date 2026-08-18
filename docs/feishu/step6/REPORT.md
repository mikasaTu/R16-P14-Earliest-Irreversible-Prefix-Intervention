# 一、结论摘要

本阶段的最终正式状态是 **BLOCKED_BY_EVENT_CONSTRUCTION**。accepted=false，novelty=N2_ORACLE_PROTOCOL_BOUNDARY_ONLY。Stage-1b 的 universal hypothesis 保持 KILLED_IMMUTABLE；Stage-2C 保持 BLOCKED_UPSTREAM_IMMUTABLE。

本阶段只研究 observed-safe prefix、policy-relative recoverability、last recoverable prefix、event-aligned handoff、cached-prefix reuse 和 oracle upper bound。它不验证物理不可逆性，也不产生可部署策略。由于上游事件构造和扰动资格门失败，所有被强制继续运行的下游矩阵都标记为 diagnostic_only，formal_positive_evidence_allowed=false；这些数据不能翻转任何 blocker，也不能形成正向证据。

# 二、范围与不可变边界

- 不可修改的历史结论：Stage-1b universal hypothesis=KILLED_IMMUTABLE；Stage-2B chunk executability=PASS、H_valid=16、actor-conditioned perturbation=BLOCKED、actor-history replay=BLOCKED、Track A/B=INCONCLUSIVE、local repair=NO_SIGNAL、operator router=NO_OPERATOR_ROUTING_SIGNAL。
- Stage-2C qualification=BLOCKED、replay contract=BLOCKED、Track A/B=INCONCLUSIVE。
- 本阶段不使用 RGB/VLA、world model、actor retraining、learned risk/replanability head、operator router training 或 π0.5；没有生成新的 idea。
- 旧阶段目录和旧 artifacts 完整保留，Stage-2D 使用独立 branch、worktree、目录和 fresh-process runtime。

# 三、代码与证据冻结

科学实现所在 branch 为 agent/stage2d-fresh-process-event-aligned-prefix-reuse。Phase-3 oracle 运行使用 source commit 141484c784be92373b3194f3c81c45a8471fa123、tree 618d2cdcff577ed5505c0a2da285c14593aae4dc。科学结果冻结提交为 019061b8a46c5ac6d315524fc3e0d3e2f481a0ff、tree=8f71e40ccb9e16791494ba253120b482ce039f62；文档同步提交与最终 origin/main commit/tree 以发布回执和最终汇报为准。不可变父为 74538ae3d9ff76f1c5c2d981da3a3c133829d73b。

完整 artifact checksum 文件为 artifacts/stage2d/SHA256SUMS，共覆盖 16283 个文件；SHA256SUMS 自身 hash=f5901565efea301f3b9666393249bbb2bcd0ac08b28a515b3a014f1c0db682f3。冻结 rule manifest hash=a8e9a32b7360cd60d8fc99aed84ae97c620be18abe209c5967be768b2dc2a334；primary decision hash=1e78f77aad5de60ff6523183607973e082591b390194bb12379d847cc007cb05；statistics hash=769af2d69f82491353819ac6a2aa5d08b40852daf8dbc72525e260858c1c7fc0。

# 四、Fresh-process branch isolation

branch isolation=PASS。每个独立的 event、prefix、arm、repeat 都在 multiprocessing spawn 新进程中执行，重新创建 LIBERO 环境，重放 anchor 前动作，重建四状态和三动作历史，重新调用 frozen ACT 并校验 chunk hash。分支之间不共享环境、wrapper、controller、observable cache、simulator、RNG 或 branch context。

隔离 gate 共 54 行，最大重建误差为 0.0；same-action CACHED_MATCHED 与 CACHED_NOQUERY 的 pre-tail signature 完全一致，contact/cause 一致，分支顺序置换一致，actor inference 没有修改环境或历史。测试也覆盖了三次精确重建、replay 错误阻断、S_obs 由 0 到 1 的阻断、禁止 global-step fallback、禁止 demonstration chunk 进入正式证据和断点续传不覆盖已完成 shard。

# 五、Init pool 与事件机会

每个选定任务生成并冻结 100 个新初始状态。0–9 只用于 infrastructure，10–39 用于 calibration，40–79 用于 evaluation，80–99 是未读取 reserve。完整 pool hash=054c2fcfa71bc3f911a7c1b2eaf5709712ba535be4ae0b599d5a955cd8b0a46a；每个 state 和 pool manifest 均已写入 artifacts。

使用 actor seed/checkpoint 7、17、29，严格去除 global-step fallback。actor event attempts 共 272 个 eligible events。关键上游 blocker 是 cream-cheese 任务的 seed 7 只有 9 个事件、distinct init IDs 只有 29，未达到每个任务的事件最低条件；因此 event_construction=BLOCKED。

# 六、扰动 qualification

target-shift 家族只移动 target bowl，方向取名义接近方向的 lateral 方向；cream cheese 从未被 teleport。三种 shift 中只有 shift_060mm 达到资格，shift_040mm 的 delayed violation=0.1273 太低，shift_080mm=0.9636 超过预注册 band。

path-obstacle 家族使用现有 blocker 放置在未来 swept-path 点，半径来自 live geometry 的保守半径和 clearance delta；不允许 injection contact，也不移动 manipulated bowl。所有 future index 和 clearance cells 均未满足资格，部分 future index=12 cells 还有 45 条 replay errors。故 target_shift_qualification=BLOCKED、path_obstacle_qualification=BLOCKED，perturbation qualification blocker 不被下游结果放宽。

# 七、Calibration 与 confirmatory 矩阵

按照外层任务“即使 gate 失败也完成剩余实验”的明确要求，继续运行了 diagnostic-only 矩阵。Phase-2 calibration matrix 为 10620/10620，error=0，missing=0；raw H1=True，但 oracle mechanism=NO_ORACLE_GAP。冻结 rule 为 target shift: clip(release_index - 2, d, H)，path obstacle: clip(predicted_path_intersection_index - 2, d, H)，tail horizon=4，实际 actor calls 只计真实调用，不用 dummy calls 补齐。

confirmatory split 40–79 的 method rows=1386/1386，replay rows=462/462，replay-admitted events=154，error=0，missing=0。H1、H2、H3、H4 均保持 INCONCLUSIVE，因为正式事件与资格门已失败；confirmatory rows 仅供机制诊断。

# 八、Oracle appendix

冻结 primary decision 和 rule 之后，追加运行 evaluation oracle appendix。154 events × 15 prefix positions=2310 branches，rows=2310/2310，events=154，error=0，missing=0，complete_matrix=true，primary_decision_effect=NONE，primary_decision_unchanged=true。terminal receipt 的科学 worker barrier 为 SUCCEEDED，oracle mechanism 仍为 NO_ORACLE_GAP；该 appendix 是 oracle upper bound，不能 retune rule、baseline、threshold 或 decision。

PAI 运行名为 r16p14-stage2d-phase3-oracle-20260819-v6，JobId=dlcxqlerueks20h6，资源为 2×A800。必须如实区分控制面状态：PAI platform final status=Failed，原因是 2310 个 oracle shard 写入后，旧 SHA256SUMS 尚未重建就先运行了 pytest，导致 test_29 在 postprocess 阶段失败。科学矩阵、terminal receipt 和所有 shard 已完整保留，不能将平台 Failed 改写为 Succeeded。随后在本地按正确发布顺序重建 checksum 并运行最终测试，得到 44 passed、0 failed、0 skipped。

# 九、机制反解（code-first audit）

机制审计基于真实 arm 代码和真实 rows，记录 new_idea_generated=false。CACHED_MATCHED 与 FRESH_MATCHED 在同一 k、相同 detection-time call、相同 tail actor、相同 horizon、相同 action budget 和相同 environment seed 下比较旧 action bytes 与 fresh action bytes；CACHED_NOQUERY 用于隔离被丢弃的查询；HOLD 用于隔离等待步数；FULL_OLD 用于观察 stale suffix 的影响。

## H2：缓存动作内容

rule_k overall 分布为 {7:82, 11:64, 10:7, 5:1}；bowl 全部 k=7，cream 多数 k=11。CACHED 与 FRESH 的 action disagreement mean=0.094（bowl=0.084，cream=0.106），说明动作内容确实不同；但 safe-success difference=0，任何真实 efficiency field 都没有达到 15%。bowl 只出现约 1.60 mm 的 EEF path 差，cream 约 0.04 mm，cause 差为 0 或轻微改善，未转化为结果价值。因此 H2 不支持 cached-prefix content value。

## H3：event-aligned handoff

arm plan 的代码结构直接解释了负向结果：IMMEDIATE_FRESH 在 d 处执行 4 个 common-tail actions 和 1 次 actor call；EVENT_ALIGNED_CACHED 先执行 k-d 个 cached actions，再执行同一 tail，并多 1 次 call。于是 event-aligned 相比 immediate 的 actual post-detection actions 增加约 6.753、cause 增加约 0.266；bowl（k=7）约增加 4.94 actions 且 cause 不变，cream（多数 k=11）约增加 8.82 actions 且 cause 增加约 0.569。这个结果是预注册执行窗口和暴露长度的机制后果，不是新的 idea。

## H4：interior selection 与 fixed baseline

与 strongest fixed baseline FIXED_DELAY_8 比较，event rule 的 safe-success 差为 -0.65 percentage points，cause 增加 8.44 percentage points，actions 减少 1.18，约 9.9% reduction，低于 15% 要求；oracle efficiency gap recovery=0.158，低于 0.40 要求。bowl k=7 约省 2.94 actions 且 cause 无差，cream 反而多约 0.82 action、cause 增加 18.06 percentage points，显示两个任务族的异质性。被丢弃的 detection-time query 只带来约 1 次 actor call 和 0.0051 秒计算开销，same-action signature 与物理 outcome 不变。

# 十、统计契约

bootstrap=10000，seed=216214。overall cluster 使用 task + source rollout/init-state；severity cluster 额外包含 perturbation parameter。同一 init-state 跨 actor seed 不当作独立样本，actor seed 作为 repeated model measurement 分层报告；prefix rows 先在 source rollout 内聚合，不能作为独立观测。报告包含 task、severity、actor seed 分层和 Holm 校正；仅有两个 task family，不做 population-wide cross-task generalization 声明。

cached-vs-fresh action-disagreement 的 global Spearman=0.266886，state-centered Spearman=0.142832，within-event positive fraction=0.653846，episode-intercept slope=0.058175，high-error AUROC 不可用。该信号是观察性诊断，不是 learned risk score，也没有引入视觉编码器或 world model。

# 十一、结果向量

- physical_group_valid：NOT_APPLICABLE（本阶段不做该验证）。
- branch_isolation：PASS。
- event_construction：BLOCKED。
- target_shift_qualification：BLOCKED。
- path_obstacle_qualification：BLOCKED。
- oracle_mechanism：NO_ORACLE_GAP。
- h1_observed_safe_window：INCONCLUSIVE。
- h2_cached_prefix_content：INCONCLUSIVE。
- h3_event_aligned_handoff：INCONCLUSIVE。
- h4_nontrivial_selection：INCONCLUSIVE。
- cached_prefix_claim：RETIRED。
- overall：BLOCKED_BY_EVENT_CONSTRUCTION。
- accepted：false。
- novelty：N2_ORACLE_PROTOCOL_BOUNDARY_ONLY。

# 十二、测试、交付与链接

Stage-2D tests 共 44 项，最终本地结果为 44 passed、0 failed、0 skipped；two-task real LIBERO integration smoke 已通过。代码、JSONL、CSV、NPZ、Markdown、测试日志、PAI receipts、2310 个 oracle shards 和 SHA256SUMS 均在仓库的 experiments/r16_p14_stage2d/ 与 artifacts/stage2d/ 下。实验计划文档已修正为一致的一级标题，并完成 full、outline、keyword 回读。

GitHub 仓库：[R16-P14-Earliest-Irreversible-Prefix-Intervention](https://github.com/mikasaTu/R16-P14-Earliest-Irreversible-Prefix-Intervention)。Step6 计划：[飞书 step6](https://icnbwz7kd1ui.feishu.cn/wiki/CoSdwjvXGixf4dkhXE5cpzUSnNY)。Step6 实验报告：[飞书实验报告](https://icnbwz7kd1ui.feishu.cn/wiki/Yqg8wGp8Fit8enkJZl9chjhVnuf)。

本阶段到此停止 offline pilot，不进入 closed-loop expansion、learned replanability、SmolVLA、π0.5、LIBERO 新扩展或其他下游阶段。
