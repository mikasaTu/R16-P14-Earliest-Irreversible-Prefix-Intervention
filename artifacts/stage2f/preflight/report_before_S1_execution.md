# R16-P14 Stage-2F / step8 实验报告

## 当前结论与完成边界

本轮执行的是 Phase 0A CPU 重分析及 S1 静态前置核验。**Stage-2F S1 全部实验尚未完成**；Phase 0B/1/2 未运行，K1/K2/K3 均未测。未提交 PAI、未启动模型训练或推理、GPU 用量 0 小时。不得将静态核验 PASS 解释为 S1 科学成功。

阻塞状态：`BLOCKED_BY_FROZEN_SPAWN_ZERO_INJECTION_CONTRACT`。它不是一个科学 K 门失败，而是用户计划要求直接复用的冻结代码必经禁用扰动路径。已完成独立可执行的 CPU 部分，没有因为某个科学 gate 阴性停止其它可执行实验。

## Phase 0A 数值

扩展后的两个 cohort 最强参考均为 `immediate_fresh_h16`。G0-1=`INCONCLUSIVE`，未建立相对该参考的稳健性能优势。全部为历史数据的 diagnostic_only CPU 重分析。

| cohort | 支撑 | event数 | kLR 事件均值 | immediate 同事件均值 | kLR cluster均值 | immediate cluster均值 | 配对 cluster Δ (pp) | 95% CI (pp) | clusters |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| all_contaminated | restricted | 21 | 38.10% | 31.75% | 34.72% | 36.57% | -1.85 | [-25.00, 17.59] | 12 |
| all_contaminated | full | 48 | 19.44% | 16.67% | 19.44% | 16.67% | +2.78 | [-5.56, 12.50] | 16 |
| replay_valid_subset | restricted | 9 | 44.44% | 37.04% | 38.89% | 46.30% | -7.41 | [-50.00, 29.63] | 6 |
| replay_valid_subset | full | 34 | 14.71% | 12.75% | 12.70% | 11.11% | +1.59 | [-6.35, 11.90] | 14 |

绝对事件均值先在事件内对三个 actor 取均值，再平均事件；统计判据使用事件→(task,init_state_id)→cluster 等权均值。两种权重不同，所以受限支撑下事件均值差可以为正而 cluster 配对差为负；上表同时给 cluster 绝对值，禁止将不同估计量相减。bootstrap=10000、seed=216214。

### 下降的代码与计数机制

冻结 S0 `_event_method_rows`（scripts/run_r16p14_stage2e_s0.py:536–549）在 restricted=True 时直接丢弃无 kLR baseline 行的事件；full support 时改用检测点 k=2 的 fresh_h16 结果。这个支撑变化把条件成功率改成覆盖所有事件的流程成功率，并未改变 actor 权重或重新运行轨迹。
- all_contaminated：受限 actor-row success 24/63=38.10%；新增 fallback actor-row success 4/81=4.94%；全支撑 actor-row success 28/144=19.44%。新增事件的低成功率直接解释绝对水平下降。
- replay_valid_subset：受限 actor-row success 12/27=44.44%；新增 fallback actor-row success 3/75=4.00%；全支撑 actor-row success 15/102=14.71%。新增事件的低成功率直接解释绝对水平下降。

这是从固定行计数和代码分支确认的**支撑选择机制**。它不是自然失败事件上的 recovery operator 因果优势，也不能证明算子族交叉。对照扩展只修正比较对象：旧 `_fixed_winner`（:594–606）排除 immediate；本次在内存 AST 中仅将其加入候选并重定向 A 的输出目录，不编辑冻结脚本。两个 full-support CI 均跨零，且下界分别 -5.56 pp/-6.35 pp，未满足原 A 路径的严格 `lower > -5 pp` 条件。因此保留 INCONCLUSIVE，不宣称显著提高、退化或完全等价。

## 冻结分支入口的机理核验

证据基线 GitHub main `508e8a5c88780066a00d994c32d673d558594867`。Stage-1b/2B/2C/2D 原判决与全部 a–e 冻结产物保持；Stage-2D 既有 fresh-process 隔离 PASS 和最大重建误差 0.0 不重新推导。

实际调用链：

1. `experiments/r16_p14_stage2d/r16_p14_stage2d/fresh_process.py:11` 从 `.runtime` 导入 `execute_branch`；第 17 行子进程调用该函数。其接口只有 event/parameter/prefix/arm/repeat/device/timeout，没有 branch backend 参数。
2. `stage2d/runtime.py:392` 在正常分支中无条件调用 `apply_perturbation`，第 395 行调用 `observe_injection`。目标移动会改目标 qpos；障碍路径会重放置障碍物。
3. `stage2c/runtime.py:381` 的 `reconstruct_to_prefix` 同样注入，第 384–385 行反而拒绝未注入路径。计划已经允许 stage2f 新写零注入 reconstruct，所以 2C 本身无需额外授权；真正冲突是必须直接复用且不得重写的 2D spawn dispatcher。
4. 零幅度仍调用禁用函数；障碍 clearance=0 仍可能将障碍放到未来路径位置。因此“给参数填零”不满足零注入，也可能改变事件源和失败机制。
5. `CauseTracker` 的 release/contact 判据被 `injected` 条件门控；删去注入调用并不能自动得到正确的自然失败 cause。需要保留 release-based 语义、用离线纯函数并列记录拓扑候选，不能把自然轨迹上永不置位的旧标签当作零风险证据。

这是静态代码确认的控制流与语义依赖，未执行扰动函数、未运行分支，也没有观测到本阶段任何跨族增益或下降。候选机制不能升级成已测恢复机制。

## 数据与仪表前置核验

三个共享多任务 ACT checkpoint seed 7/17/29 均存在，hash 与冻结 manifest 一致，单个约 12.6 MB，归属 2254:2254，覆盖所需两任务。记录见 `artifacts/stage2f/preflight/actor_assets.json`，这只是文件存在/hash 证据，不是本轮运行成功。

既有 Stage-2D pool 的第二任务为 stove 且 split 不同，因此不能替代 S1 pool。Phase 0B.1 已授权创建两个任务各 100 个新 reset-randomized state；这本来是任务内容，不能把尚未创建误报成外部缺数据。

冻结 Stage-2D 的状态点 contact signature 不是 D1 所需逐 simulation step 的完整原始 contact 记录；原有采集条件还含扰动任务专用限制。后续必须在 stage2f 实现新 collector、纯 cause 标签及显式 pid/env_hash/chunk_hash，而不改 a–e。

## K 门、绝对水平与未执行矩阵

| 项目 | 预注册判据 | 本轮实测 | 判决 |
|---|---|---|---|
| K1 | 两任务各 calibration≥25 且 evaluation≥25 合格事件 | 无事件采集，不是零事件产率 | NOT_EVALUATED |
| K2 | 同一预算在两任务 oracle_best∈[0.25,0.85]，arm gap≥0.15 | 无 21,600 行预算扫描，预算未选择 | NOT_EVALUATED |
| K3-a | 两族均定义事件≥30/任务 | None | NOT_EVALUATED |
| K3-b | minority crossing≥0.15；CI 下界>族内 null 95分位 | crossing/CI/null 均未测 | NOT_EVALUATED |
| K3-c | Spearman≤0.8/任务 | None | NOT_EVALUATED |
| U_A/U_B 平均 safe success | 分别报告，不能仅报告差值 | 两者均未测 | NOT_RUN |

Phase 0B/1/2 的 summary.json 是明确的未运行状态收据；没有伪造 events、grid_rows、atlas、boundaries、crossing 或 null_distribution。没有 selection receipt，因此 evaluation outcome 继续 open-deny。当前没有 NONTRIVIAL_OPERATOR_RELATIVITY 或 MONOTONE_RESCALING_ONLY 的科学判决。

## 资源、停机窗口与恢复

资源保守采用附件更具体的限制：最多两个作业、每作业≤2 A800、总预算≤20 GPU 小时；单节点 CPU≤88 core、memory≤正文 1.4 TB。用户希望使用 robot 下闲时容量，具体资源 alias 与实际 idle placement 尚未做正式提交验证。

PAI registry 已只读检查可用命令和 DLC binary 固定 hash；凭据仅查 numeric owner/mode，未读取密钥；未做 W&B live membership 验证，未宣称预提交合格。控制面记录见 `preflight/controller_inventory.json`。作业清单 `pai/jobs.json` 为空；run_id/job_id/tree hash/final status 没有可填写的运行行，不能用本地工作树冒充 PAI JobId。

北京时间 09:30–09:40、19:30–19:40 禁止本工作流作业。当前没有本轮作业，故不会占用窗口；尚未安装定时停机服务，不能声称已有自动停机保障。未来任何提交之前，必须先落实精确 JobId 的定时 StopJob 与窗口后持久分片恢复。没有前驱替代任务，cleanup targets=0；未删除任何 PAI 记录或 CPFS 证据。

## 本阶段没有测的东西

未测 S1 自然失败产率、预算工作区、非嵌套算子族边界交叉、seed 噪声地板、S1 safe-success 绝对水平或任何真实机器人效果。未重训 ACT，未引入 VLA/世界模型/RGB，未注入任何扰动，未添加新 idea，未自行启动 S2。Phase 0A 使用历史受扰动/受支撑选择影响的数据，只能解释旧表，不证明自然事件上的结论。

## 继续执行所需的最小计划修订

需要明确允许 **仅在 stage2f 新增独立 spawn dispatcher 和零注入 branch backend**，保持旧 `fresh_process.py`、`isolation.py`、2C/2D runtime 完全冻结，并沿用其独立进程及签名测量要求。不能用 monkeypatch、修改旧模块或传零幅度代替这个修订。CPU 重分析不依赖此项，已单独推进。

另外，PREREG 已写明全部 K 数值门槛；policy-call cap、跨任务预算排序聚合、split-half 小样本退化处理等执行细则尚未冻结为可执行配置。它们只能在任何 S1 数据打开之前登记单独补充，不得事后根据 outcome 补选。当前预注册不是 GPU ready 声明。

## 复现、校验与发布

CPU 复算：

```bash
PYTHONDONTWRITEBYTECODE=1 /mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/libero_sim/bin/python experiments/r16_p14_stage2f/phase0a.py
python3 scripts/verify_r16p14_stage2f.py
```

第一条需要 NumPy，实际使用现有 libero_sim Python 环境；没有 simulator/model execution。第二条只用标准库，在完整克隆中验证 SHA256 覆盖、a–e 冻结路径、输入 hash、静态禁用调用链及未运行阶段的明确状态。`phase0a/tests.json` 记录候选判别测试、四个输入 hash/schema 和九个 A1 表项精确复现；`preflight/static_checks.json` 记录静态语法与调用链核验；最终验证收据见 `artifacts/stage2f/verification.json`。

所有文件由 `artifacts/stage2f/SHA256SUMS` 绑定；该清单本身由 Git commit 绑定，不做不可成立的自哈希。PREREG 在零 S1 rollout 状态下单独提交，然后再提交报告及校验清单。发布目标是 GitHub `mikasaTu/R16-P14-Earliest-Irreversible-Prefix-Intervention` main；飞书 step8 和实验报告 token 见 `feishu_documents.json`。最终 commit 与远端回读证据以飞书发布回执和主任务答复为准。
