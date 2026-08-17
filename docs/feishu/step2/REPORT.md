<title>实验报告</title>

# R16-P14 Stage-1b：A+B 首次检查点

日期：2026-08-13 UTC。当前报告严格区分指标修正与 replay correctness；它不是 learned detector、可部署 intervention policy 或最终算法的有效性证据。

## 结论摘要

- **Phase A：旧 Bowl 正信号不成立。**修正 cause_violation 与 retention 定义后，Bowl 的 median window 从 4 降为 0，median post-detection retention 从 50% 降为 0%。
- **Phase B：Replay gate 通过。**fresh-env prefix reconstruction 在 180/180 个分支点完全一致，contact/outcome 180/180 一致，最大终态误差为 0。
- 旧 snapshot-restore 对照仅 147/180 个分支点通过，contact/outcome 为 149/180，最大终态误差为 3.7161693483379317。
- 没有 replay 或基础设施 blocker；下一步只能进入不使用 BC 的 expert action-chunk calibration。

## 1. 冻结边界

- 当前 main freeze：commit ee56aa096e308214c38132d0e6d2a9e576c29792；tree 50e56b84c5e867447b22f81e31454646a12a9eb8。
- Stage-1 frozen source：commit a1b61194a8382f5b1a247b9cd9b140645ff2aeb8；tree 53001c43fbbb165c0a1f2c71f9cbd4c81b9d0ced。
- 独立 branch/worktree：agent/stage1b-contract-repair，/workspace/leon/R16-P14-stage1b。
- Stage-1 source 与原 artifacts 未修改。

## 2. Phase A：90 条记录的修正复算

Phase A 离线重读全部 90 条 immutable Stage-1 candidate records，没有启动 simulator 或 GPU。cause_violation 现在只统计明确的目标 cause；task_failure、timeout、safe_success 分开保存。

### 旧指标与修正指标

- Drawer：old mixed unsafe 27；explicit cause violation 6；failure-only inclusion 21；corrected median window 0；retention 0%；完整 M0/M1/M2 pair 为 0。
- Bowl：old mixed unsafe 17；explicit cause violation 5；failure-only inclusion 12；old median window 4，corrected median window 0；old retention 50%，corrected post-detection retention 0%；完整 pair 为 2。
- Wine：old mixed unsafe 28；explicit cause violation 28；corrected median window 0；retention 0%；完整 pair 为 4。

修正 retention 主定义为 (k_last_safe - d) / (chunk_length - d)。旧实现使用 k_last_safe / chunk_length，并把预算内未完成任务混入 safety violation。

### Bowl 正信号判定

**不再存在。**只有 5/30 个 Bowl candidates 具有显式 nominal cause violation，且仅 2 个具有完整 M0/M1/M2 比较。这两个 pair 的 cause-violation 计数分别为 0/1/1，因此 intervention 没有在 safety 上超过 immediate full replan。

M1 在这个仅有两个 pair 的子集上，detection 后 newly executed non-nominal actions 的中位数为 41.5，M0 为 60；但样本极小且 M1 safety 更差，只能记录为弱 rework observation，不能视为正结果。

## 3. Phase B：Bowl replay reconstruction repair

repaired initializer 使用 fresh independent environment，恢复 demonstration phase anchor，精确重放 nominal prefix，在 d 注入已记录 perturbation，再继续至指定 branch point。每个 operator/repetition 均不复用执行过的 mutable environment。

### Snapshot restore 对比 prefix reconstruction

- Published Stage-1 snapshot restore：insertion pass 26/30，即 86.7%。
- Snapshot-restore control，5 repeats：insertion pass 26/30；all branch-point pass 147/180，即 81.7%；contact/outcome 149/180，即 82.8%；max error 3.7161693483379317。
- Fresh-env prefix reconstruction，5 repeats：insertion pass 30/30；all branch-point pass 180/180；contact/outcome 180/180；max error 0。

只有 57/180 个 reconstructed branch hashes 等于旧 Stage-1 snapshot hash。这表明旧 initializer 丢失了 trajectory/controller history，是对照问题而非新 reconstruction contract 失败。

### Replay gate

**PASS。**预注册门禁要求 overall pass rate 不低于 99%，contact/outcome 100%，final-state max absolute error 不高于 1e-9；正式结果分别为 100%、100%、0。30/30 candidate 的 Stage-1 exact action-chunk hash 全部匹配。

仅为保持 Stage-1 frozen MLP 动作字节一致，使用一张本地 A800 执行 PyTorch 2.5.1+cu124 frozen forward；simulation/replay execution 在 CPU。没有训练模型。

## 4. 产物

- 实验 contract：preregistration.yaml、metric_contract.md、source_manifest.json、decision.json、commands.sh。
- Phase A：reanalysis.jsonl、summary.json、paired_metrics.csv、old_to_new_metrics.csv、report.md。
- Phase B：branch_reconstructions.jsonl、summary.json、report.md。
- 代码与测试：offline_reanalysis.py、replay_reconstruction.py 及对应 pytest。
- 本检查点：reports/FIRST_CHECKPOINT.md。

## 5. Blocker 与下一步

没有 replay 或 infrastructure blocker。指标修正后的科学信号目前为负，因此只能按预注册流程进入 disjoint expert-action audit；Phase D/E 仍未获准。

**下一条 scientific command：**

cd /workspace/leon/R16-P14-stage1b

R16P14_PYTHON=/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/libero_sft/bin/python MUJOCO_GL=egl PYOPENGL_PLATFORM=egl ./experiments/r16_p14_libero_stage1b/commands.sh phase-c-calibrate

该命令只允许在 demo IDs 0–9 上冻结 revised environment-only perturbation，不训练模型，不得用 evaluation 或 held-out demos 调参。

# Phase C 与最终决策

**最终决策：KILL_CORE_HYPOTHESIS。**

Stage-1b 修复了 metric 与 replay contract，但 expert action-chunk 机制未能在至少两个任务上成立。该结论只否定本轮预注册的跨任务 core hypothesis，不代表所有 prefix intervention 都无效。

## 6. Phase C：完全 policy-free 的 expert action chunk

Phase C 只使用 demonstration actions[t:t+16]，不使用 BC、policy checkpoint、learned risk、RGB 或 privileged perturbation metadata。Calibration 只读取 demo IDs 0–9；evaluation IDs 10–39 与 held-out IDs 40–49 均未读取。

### 任务与 bounded replacement

- Bowl：保留主任务，只移动 target plate，不移动 held bowl。
- Drawer：重新标定无注入接触的 obstacle placement。
- Wine-rack：rack 是静态模型几何，没有可移动 target joint；直接 teleport manipulated bottle 又被本轮禁止，因此标记不适用。
- 唯一一次 bounded replacement：LIBERO-GOAL task 6，put_the_cream_cheese_in_the_bowl。它在 calibration 前按 suite order 规则选定，具有 stable grasp、late release、movable target bowl。

在正式 grid 前记录了两项 amendment，只使用 demos 0–1 的 injection phase/contact smoke，未查看 recovery gain，未读取 evaluation 或 held-out demos：把 target shift 提前到已有 target contact 之前；fixed-magnitude x shift 的方向固定为远离 held object。完整原因与边界已写入 preregistration.yaml。

### 正式 calibration 规模

- 44 个 fixed configurations：Bowl 16、Drawer 12、replacement 16。
- 440 条 nominal calibration records。
- 59 个 cause-positive candidates 完成 prefix-stride-1 audit。
- 885/885 个 independent prefix reconstructions 完全一致。
- 每个 branch 先 hard reset，再 restore demo anchor、重放 exact expert prefix，operator 之间不共享 mutable state。

合格 contract：injection-instant violation 不高于 10%，delayed nominal violation 在 30%–80%，实际 median recoverable window 在 2–8，replay 不低于 99%。Violation onset delay 不得替代真实 recovery window。

### Task-level calibration 结果

- **Bowl：0/16 configurations qualified。**没有配置同时满足十个 demo 的 phase validity、immediate contact、rate 与实际 window contract。
- **Drawer：0/12 configurations qualified。**最接近的 offset+00 / clearance35mm 配置为 0% immediate、30% delayed、100% replay，但实际 median recoverable window=0。
- **Cream cheese in bowl：3/16 configurations qualified。**按冻结规则选择 lead=8 / target shift magnitude=40mm；0% immediate、60% delayed、median window=7、median post-detection retention=50%。

### Selected replacement 的 M0/M1/M2（calibration only，n=6）

- M0 immediate full replan：cause violation 1/6，safe success 5/6，median new non-nominal actions=34。
- M1 continue-to-last-safe then same full replan：cause violation 1/6，safe success 5/6，median new actions=33.5。
- M2 same-trigger cause-specific local repair：cause violation 3/6，safe success 2/6，median new actions=7。

Timing comparison 非空，但 M0 与 M1 aggregate safety 完全相同，new actions 只差 0.5。M2 动作更少但 safety 显著更差，因此没有 operator win。Oracle best-of-three 可在 6/6 calibration pairs 上 safe-success，但这是 privileged calibration observation，不能修复跨任务 gate。

## 7. Expert gate 与停止规则

**FAIL。**只有 1 个任务具有任何 preregistration-qualified perturbation，而 expert gate 要求至少 2 个任务 median window≥2 且 retention≥30%。Calibration parameters 必须在 evaluation 前冻结，因此单一 qualified task 的 evaluation 不可能产生第二个 qualified task；继续调 failed tasks 会违反协议。

因此 failure 在 calibration 阶段已经 monotonic，按预注册停止规则写入 KILL_CORE_HYPOTHESIS。Evaluation/held-out split 保持未读。

## 8. Phase D/E 执行状态

- Phase D policy substrate：未执行。没有 PAI job、没有新模型训练、没有 policy inference。
- Phase E revised policy oracle：未执行。没有 policy-seed oracle records 或 bootstrap CI。
- Learned/deployable evidence：不存在。

## 9. 六个科学问题的回答

1. 修正指标后 Stage-1 正信号是否存在？**否。**Bowl window 与 retention 均变为 0。
2. Bowl replay<99% 能否修复？**能。**prefix reconstruction 达到 180/180，终态误差 0。
3. Expert chunk 是否有 detection→safe prefix→no-return 窗口？单一 replacement calibration task 有；要求的至少两个任务没有。
4. 等待 last-safe 是否比 immediate replan 减少返工？没有有意义的证据；33.5 vs 34 new actions，aggregate safety 相同。
5. Same-trigger local repair 是否优于 full replan？**否。**M2 safety 更差。
6. 更强小型 policy 能否复现？未测试；Phase C stop rule 禁止进入 Phase D。

## 10. 交付物

- GitHub repository：[R16-P14-Earliest-Irreversible-Prefix-Intervention](https://github.com/mikasaTu/R16-P14-Earliest-Irreversible-Prefix-Intervention)。
- Final report：experiments/r16_p14_libero_stage1b/reports/REPORT.md。
- Raw outputs：artifacts/stage1b/offline_reanalysis、replay_gate、expert_chunk_calibration。
- Paired metrics：selected_config_paired_metrics.csv。
- Test result：artifacts/stage1b/test_results/pytest.xml。
- Integrity：artifacts/stage1b/SHA256SUMS。

## 11. 证据分层

- Metric correction：旧 Bowl claim 为负。
- Replay correctness：修复并通过。
- Expert mechanism feasibility：仅单任务 calibration signal；跨任务 gate 失败。
- Policy substrate adequacy：未知，因 gate 未到达。
- Oracle mechanism result：未运行 revised policy oracle。
- Learned/deployable evidence：本轮不存在。

## 12. GitHub 发布确认

**main commit：**e29e3ead42fd1799b412a4968e6a67aac3784874。已通过 SSH push，并用 git ls-remote 确认远端 refs/heads/main 与本地 main 精确一致。

[查看 GitHub commit](https://github.com/mikasaTu/R16-P14-Earliest-Irreversible-Prefix-Intervention/commit/e29e3ead42fd1799b412a4968e6a67aac3784874)
