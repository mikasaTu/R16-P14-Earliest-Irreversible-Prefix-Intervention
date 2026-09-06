# R16-P14 Stage-2F / step8 实验报告

更新时间：2026-09-07，北京时间。当前为执行中的阶段报告，尚未完成全部实验。

## 当前结论

Phase0A 已完成：补入 immediate_fresh_h16 后，G0-1 为 INCONCLUSIVE。S1 的 200 个固定初始状态已生成并验证；一条 infrastructure clean episode 已完成，包含 360 个控制步、9000 个物理步和完整接触拓扑。正式采集首批两作业已提交并确认UseOversoldResource=true，但由于跨节点CPFS控制文件读取发生ESTALE，已全部停止，尚无正式episode。修复重提中；K1、K2、K3均未判决。

原冻结 Stage2D 执行入口存在隐式注入路径，其历史 BLOCKED 事实保留。按照用户后续明确授权，在 Stage2F 新增零注入后端；没有改写 Stage2A–E，没有调用注入函数。新的独立分支后端已完成ROOT-first同源LIBERO的真实基础设施检查：4core+4reference+重复分支，anchor重建误差0，D1每控制步25个物理步，测量有无的终态/历史hash一致，预算与pid/env/chunk校验全部通过。首轮资源路径失败原样保留；这些是工程验证，不能计作科学成功。

## 冻结协议及执行边界

PREREG_S1 首次提交为 8fd4e9f8497350534217e417b317839264d96d70，数值门槛保持不变。执行补充协议 AMENDMENT_S1_EXECUTION.md 已在首次 S1 数据前以 164aea79 提交并推送 main；固定 init seed、full-episode failure、算子实际预算和 selection 封存规则见该文件。

按用户最新指示，科学 gate 不通过不取消其他独立实验；缺失前提不会被补造。若 K2 没有合法预算，不会擅自选预算打开 evaluation。资源上限使用计划中的更严格限制：最多2个 PAI 作业，每个2张 A800，总计20 GPU小时。北京时间09:30–09:40、19:30–19:40无本任务作业；控制器提前5分钟关停和拒绝提交，:40后恢复已完成分片。闲时资格须从 exact JobId ListJobs 验证 UseOversoldResource=true。

## Phase0A：绝对水平与机制

| 支持集 | k_last_recoverable event mean | immediate event mean | paired cluster delta | 95% CI |
|---|---:|---:|---:|---:|
| 全部事件，restricted | 38.0952% | 31.7460% | -1.8519 pp | [-25.0000,17.5926] pp |
| 全部事件，full | 19.4444% | 16.6667% | +2.7778 pp | [-5.5556,12.5000] pp |
| replay-valid，restricted | 44.4444% | 37.0370% | -7.4074 pp | [-50.0000,29.6296] pp |
| replay-valid，full | 14.7059% | 12.7451% | +1.5873 pp | [-6.3492,11.9048] pp |

event mean 与 cluster mean 不可混用。38.1%→19.4% 的分解为 restricted 24/63，加上未定义事件 fallback 4/81，得到 full 28/144。模型没有重新训练或改变；支持规则扩展改变了分母。加入更强的 immediate comparator 也缩小了相对差距。这些比较全为 diagnostic-only，不支持性能优势。完整代码机制与计数审计见 MECHANISM_PHASE0A.md。

## S1 进度与尚无的结果

- Phase0B：两任务各100个唯一 reset state；ID划分保持0–9/10–49/50–89/90–99。infrastructure v1 原记录及hash问题保存在 preflight/infra_smoke_v1，正式采集重新执行600条。
- K1：calibration/evaluation 合格自然失败事件正式计数尚未获得，未判决。
- Phase1 / K2：9个配置预算档、4算子、3 recovery seeds、20 calibration events/任务尚未执行完；没有 selection。
- Phase2 / K3：未打开 evaluation；两族平均 safe success、k*、跨族minority crossing、Spearman以及族内split-half噪声地板均尚无可报告结果。
- 静态PAI测试：新2GPU资源合同4项、提交黑窗与待提交cap5项、canonical回归25项及7subtests通过。统计/汇总/后端20项测试通过；两个任务init0在正式spawn启动路径下逐字段复现。真实后端11rows与9条trace已由主线程逐项验收，源码SHA为999901d941ac107d29c11fd0e0f4dbf2587470794b779b5cb8353681b4267e81；CPFS读取3项测试、自动resume9项测试通过。

## 本阶段没有测的东西

没有训练 actor，没有 VLA、世界模型、RGB策略、扰动注入或真实机器人评测；不测性能赢面，不重新验证冻结的 universal hypothesis，不提出新idea，不启动S2。没有把代码测试、基础设施 episode 或 Running 状态写成科学成功。

## PAI 首批执行记录

- dlcfesoi2j9tp8y5（cream，r2）：Stopped，exact idle=true，2254身份；读取控制heartbeat触发ESTALE，0正式episode。
- dlc189mgayv5nwjf（bowl，r1）：Stopped，exact idle=true，2254身份；收到全局停止标记后退出，0正式episode。
- 两者源commit94d46b86、tree306953eabdc5a3c2680551c8078c1b67c92c2e05。失败FATAL_ERROR和最终readback已保存到artifacts/stage2f/pai_jobs。新版本为可恢复的CPFS短暂inode替换增加有界重读；持续错误仍停止。黑窗及20GPU小时上限保持。
