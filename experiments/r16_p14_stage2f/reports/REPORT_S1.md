# R16-P14 Stage-2F / step8 实验报告

## 当前结论与执行状态

Phase0A 和 Phase0B 已完成。Phase0B 的两任务共600条正式 clean rollout 均已落盘，两个PAI采集作业均为 Succeeded，持久化完整性和2254:2254归属已核验。K1不通过：cream的calibration/evaluation合格事件为19/23，bowl为11/13，均低于每侧25个门槛。

按用户正文“不因gate失败停止其他实验”的要求，继续现有30个calibration事件上的全部9个预算、3个恢复seed、4个算子、5个前缀及4个参考臂，共19440个请求。原计划每任务20个事件的要求保留，实际缺10个，不补样、不换split。该网格只能报告现有观测下的结果与缺口；不足计划样本不能形成有效K2 selection。Phase2 evaluation仍为open-deny，K3未测，不能据此判定NONTRIVIAL_OPERATOR_RELATIVITY或MONOTONE_RESCALING_ONLY。

## 冻结协议与运行身份

PREREG首次提交8fd4e9f8497350534217e417b317839264d96d70；K1/K2/K3数值不变。零注入执行补充164aea79在首次S1数据前封存。现有calibration样本网格的执行说明EXECUTION_CONTINUATION_AVAILABLE_CALIBRATION.md在首次Phase1恢复结果之前封存，明确样本缺口和selection禁止规则。

- 采集源码：[6b788a0764904e11e022c4330a74fa3e009c9a33](https://github.com/mikasaTu/R16-P14-Earliest-Irreversible-Prefix-Intervention/commit/6b788a0764904e11e022c4330a74fa3e009c9a33)，tree f13e12a2b006632e48e5aa64ac2c53e30928de51。
- Phase1网格源码及K1证据：[a0888d751117cbf7c5a73080d1ee1f421689e8bf](https://github.com/mikasaTu/R16-P14-Earliest-Irreversible-Prefix-Intervention/commit/a0888d751117cbf7c5a73080d1ee1f421689e8bf)，已推送GitHub main并回读。
- 运行环境：CPython3.11.11、torch2.6.0+cu124、NumPy2.4.6、MuJoCo3.6.0、robosuite1.4.0、wandb0.27.2；实际PAI节点A800-SXM4-80GB、driver550.54.15。独立兼容环境保留原共享环境，启动核对实际导入路径、版本、SHA和绑定。
- 固定shared multitask ACT seed7/17/29 checkpoint，无再训练；固定200个init state及0–9/10–49/50–89/90–99划分，无重采样。
- 每个恢复branch独立spawn进程和fresh env；冻结Stage2A–E未修改。Stage2F零注入入口只复用安全重建和纯动作函数，没有执行扰动、目标移动或blocker路径。

## Phase0A：绝对水平与机制

| 支持集 | k_last_recoverable event mean | immediate event mean | paired cluster delta | 95% CI |
|---|---:|---:|---:|---:|
| 全部事件，restricted | 38.0952% | 31.7460% | -1.8519 pp | [-25.0000,17.5926] pp |
| 全部事件，full | 19.4444% | 16.6667% | +2.7778 pp | [-5.5556,12.5000] pp |
| replay-valid，restricted | 44.4444% | 37.0370% | -7.4074 pp | [-50.0000,29.6296] pp |
| replay-valid，full | 14.7059% | 12.7451% | +1.5873 pp | [-6.3492,11.9048] pp |

G0-1为INCONCLUSIVE。event mean与cluster mean的权重不同，不能混用。38.1%→19.4%可逐项分解为restricted的24/63，加上未定义事件fallback的4/81，得到full的28/144；模型没有改变，支持规则扩展改变了分母。加入immediate comparator也缩小相对差距，现有证据不支持性能赢面。

另一个易混淆的计数：冻结Stage2C evaluation的immediate为23/144，而本表calibration为24/144。两者虽同分母，init和split不同；这不是代码带来的“多成功1次”，也不是fallback改善。完整逐行计数和代码机制见MECHANISM_PHASE0A.md。全部Phase0A结果为diagnostic-only。

## Phase0B与K1最终判决

| 任务 | 正式rollout | infrastructure合格/30 | calibration合格/120 | evaluation合格/120 | reserve合格/30 | K1 |
|---|---:|---:|---:|---:|---:|---|
| cream cheese in bowl | 300 | 5 | 19 | 23 | 2 | BLOCKED_BY_NATURAL_EVENT_YIELD |
| bowl on plate | 300 | 4 | 11 | 13 | 2 | BLOCKED_BY_NATURAL_EVENT_YIELD |

合格事件定义为首次结构anchor（grasp、stable lift≥2、gripper closed、尚未成功、chunk16）之后，完整clean episode最终失败；每条episode至多一个事件。K1只使用公开qualification元数据计数。

600个固定(task,init,seed)键完整、唯一。360条非evaluation episode的源提交、runtime receipt和trace SHA已核验并发布；240条evaluation episode及对应trace仅做文件哈希/归属核验，保持密封，不读取或解释clean outcome。完整接触拓扑同时保存raw geom pairs和规范化排序集合，release-based与prerelease topology候选标签分开记录。

calibration 的120条固定rollout可完整分解：cream为53条clean success、48条完整horizon失败但无结构anchor、19条合格自然失败；bowl为51、58、11。无anchor失败和已成功episode都被预定事件定义排除，因此事件不足不来自release/topology标签的额外筛选。现有元数据不能进一步判定无anchor到底缺grasp、两步lift还是gripper条件，细分保持unknown。逐seed计数、四条真实trace核验和代码入口见MECHANISM_PHASE0B.md。

## Phase1、K2与Phase2边界

| 任务 | 可用/计划calibration事件 | core请求 | reference请求 | 总请求 |
|---|---:|---:|---:|---:|
| cream cheese in bowl | 19/20 | 10260 | 2052 | 12312 |
| bowl on plate | 11/20 | 5940 | 1188 | 7128 |
| 合计 | 30/40 | 16200 | 3240 | 19440 |

请求覆盖tail_horizon={4,8,16}、action_budget={8,16,32}、recovery policy call cap=8，另有1次generator validation调用（总调用最多9）；四算子与全部参考臂保持原定义。现有请求完整执行和计划样本完整是两个不同字段，结果必须报告planned_sample_complete与missing_planned_events。

bowl的一个calibration事件anchor位于316/320步，k8/12/16及对应部分参考请求超出任务horizon。这产生324个core与54个reference请求的结构性BLOCKED。完整事件支持集的描述性指标排除该事件，实际支持为cream19、bowl10，但仍保留原始30事件、19440请求和全部排除理由。这些组合记录BLOCKED并继续其余请求；不截短缓存前缀以伪造计划执行，也不把BLOCKED的占位False当阴性观测。每档oracle、weakest、gap及有效支持集将在实际分片完成后报告。

两个Phase1作业dlc14obea2cugtbf与dlc4vt6h6pcw6jzk已提交；两个任务的实际分支源提交、runtime receipt、预算、重建误差0和trace SHA已通过主线程验收，完整网格仍在运行。北京时间2026-09-07 04:23的快照为cream3156/12312、bowl2846/7128，两个作业均无FATAL_ERROR。目前没有可报告的最终Phase1恢复成功率。K2需两任务在同一档位同时满足oracle∈[0.25,0.85]且gap≥0.15，并满足完整证据要求；计划样本不足使有效selection不可成立。Phase2所需两族safe success、k*、minority crossing、Spearman、10000次cluster bootstrap以及族内split-half噪声地板均未测，不填0、不编造边界或概念判决。

## PAI记录与工程修复

| JobId | 工作 | 状态/结果 |
|---|---|---|
| dlcfesoi2j9tp8y5 | 旧cream采集 | Stopped；CPFS ESTALE，0正式episode |
| dlc189mgayv5nwjf | 旧bowl采集 | Stopped；全局停止，0正式episode |
| dlcwm6bd6bomjggq | 旧cream采集 | Stopped；torch cu130与driver550不兼容，0正式episode |
| dlc1qayvbetb16ji | 旧bowl采集 | Stopped；同轮环境修复停机，0正式episode |
| dlc1hzmadm185c68 | CUDA12.4 cream采集 | Succeeded；300/300持久化 |
| dlc1rz7o5mf1vajn | CUDA12.4 bowl采集 | Succeeded；300/300持久化 |
| dlc14obea2cugtbf | cream现有样本网格 | Running；已有真实分支 |
| dlc4vt6h6pcw6jzk | bowl现有样本网格 | Running；已有真实分支 |

全部已提交作业均逐JobId核验UseOversoldResource=true。失败日志、实际环境、最终状态和源tree原样保留。正式任务最多2个、每个2×A800/24CPU/200Gi；总体20GPU小时上限，控制器19.8小时提前停止。600条采集完成时累计保守上界约1.30814 GPU小时（包含失败启动、排队及dev14检查预留）。

北京时间09:25/19:25开始拒绝提交并停止本任务精确JobId，以覆盖09:30–09:40和19:30–19:40禁跑窗；:40后按持久化分片恢复。控制器和resumer对CPFS短暂读取错误有界重试；持续控制I/O失败拒绝提交并尝试停止已知活跃作业，Stop失败继续重试。CreateJob本身不会盲目重试。

兼容环境单GPU工程检查完成360步clean episode及独立fresh_h4分支，重建误差0、D1物理步与预算校验通过；真实raw trace由主线程复核。旧环境基础设施episode不与正式事件混用。CPU回归、矩阵短缺/越界继续执行、提交与恢复I/O检查均有独立receipt；验证脚本PASS仅表示已发布证据完整性，不表示科学主张成功。

实际算子预算经同事件、同k、同recovery seed和同预算的四臂配对核验：tail4/action32时四臂均执行32个新恢复动作，8次recovery policy调用加1次validation。hold/rollback的单步前导动作占用相同action budget，因此其余policy动作是31步，fresh为32步。算子名字中的h16还会受configured tail_horizon截断；不能只根据名字声称h16始终16步重规划。27个完整跨预算/seed配对组（每组108行）的detection和pre-tail状态一致，未发现D4错配；这些是工程验收而非恢复成功率结论。

当前CPU全套回归为61 passed，覆盖样本短缺、源事件/参考臂缺失、真/伪horizon越界和selection禁止规则；旧测试回执保留对应历史代码，不与当前测试重复累加。

## 本阶段没有测的东西

没有训练actor，没有VLA、世界模型、RGB策略、扰动注入或真实机器人评测；不再测性能赢面，不重新验证冻结的universal hypothesis，不提出新idea，不启动S2。K1失败不能代替K3判决；现有样本下的网格结果也不能冒充原计划样本完整的确认性结论。
