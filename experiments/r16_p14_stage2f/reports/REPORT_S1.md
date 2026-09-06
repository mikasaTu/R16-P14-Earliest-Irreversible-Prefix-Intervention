# R16-P14 Stage-2F / step8 实验报告

## 当前结论

Phase0A、600条Phase0B clean rollout、全部19440条Phase1可用请求已执行。两个Phase1 r2作业均为Succeeded，完整键、源身份、压缩文件SHA和D4分组验收PASS；逐物理步的完整trace验收也已PASS；整个S1仍需完成Phase2。

K1不通过：cream的calibration/evaluation合格事件为19/23，bowl为11/13，均低于每侧25。K2九档无共同工作区：bowl最高oracle-best为22.6667%，对应最大gap为13.3333pp，低于25%和15pp门槛；原计划每任务20个calibration事件的样本要求也未满足。

按用户正文“不能因为一个gate不到就停止验证其他实验”继续独立诊断Phase2。正式selection receipt保留BLOCKED/null；在部分calibration数据之后封存的DIAGNOSTIC_ATLAS_CONTINUATION.md规定，仅按原calibration排序选择独立诊断预算，发布并验证真实Git提交后才打开evaluation。该补充不是事前预注册，不能冒充正式K1/K2通过。当前evaluation仍未打开，K3尚未测；不启动S2。

## 冻结设计与运行身份

- Stage1b universal hypothesis的KILLED_IMMUTABLE判决、Stage2B/C/D的冻结阻塞判决、Stage2E-S0的diagnostic_only属性保持不变；冻结Stage2A–E文件未修改。
- PREREG首次提交8fd4e9f8497350534217e417b317839264d96d70，SHA256 37bb49c77062ee91e2196f35bd2f4987b9dfa403bf3081daab6da2a850f68ec5。零注入执行补充164aea79在S1数据前封存；K1/K2/K3数值未改。
- 固定两任务、200个init state、seed7/17/29 shared multitask ACT checkpoint，无训练、无重采样、无split调整。ID0–9 infrastructure、10–49 calibration、50–89 evaluation、90–99 reserve。
- 采集源码6b788a0764904e11e022c4330a74fa3e009c9a33；Phase1 r1源码a0888d751117cbf7c5a73080d1ee1f421689e8bf；r2源码93872b41ad48d24e1c6cb46d8c46690dae359b62，tree9f25ad3ec3e98d7ce378a46fd163dfbe383d6c3d。
- CPython3.11.11、torch2.6.0+cu124、NumPy2.4.6、MuJoCo3.6.0、robosuite1.4.0，实际A800-SXM4-80GB/driver550.54.15。runtime receipt SHA256为3d7b252885b4047fb93140c305a3ef4a1d3ec9aa450d90341befd2b7e26a5e44。
- 每个branch独立spawn/fresh env，按原始reset及历史重建；重建误差阈值与D1–D5验收保留。Stage2F零注入入口未调用perturbation、target shift或blocker路径。

## Phase0A：绝对水平与机制

| 支持集 | kLR事件均值 | immediate事件均值 | 配对cluster差 | 95% CI |
|---|---:|---:|---:|---:|
| 全事件 restricted | 38.0952% | 31.7460% | -1.8519pp | [-25.0000,17.5926]pp |
| 全事件 full | 19.4444% | 16.6667% | +2.7778pp | [-5.5556,12.5000]pp |
| replay-valid restricted | 44.4444% | 37.0370% | -7.4074pp | [-50.0000,29.6296]pp |
| replay-valid full | 14.7059% | 12.7451% | +1.5873pp | [-6.3492,11.9048]pp |

G0-1=INCONCLUSIVE。事件均值与cluster均值权重不同，不能相减混用。38.1%→19.4%由restricted的24/63加上undefined fallback的4/81得到28/144；模型没变，是支持集与分母改变。冻结Stage2C evaluation的immediate23/144与本次calibration24/144也不是同一支持集上的提升。加入immediate参考后，现有证据没有建立性能赢面。本阶段不再主张优于baseline。详见MECHANISM_PHASE0A.md。

## Phase0B与K1

| 任务 | clean rollout | infrastructure/30 | calibration/120 | evaluation/120 | reserve/30 |
|---|---:|---:|---:|---:|---:|
| cream | 300 | 5 | 19 | 23 | 2 |
| bowl | 300 | 4 | 11 | 13 | 2 |

600个固定(task,init,seed)键完整且唯一。事件定义是首次满足grasp、stable lift≥2、closed gripper、task尚未成功、chunk16的anchor之后，完整clean episode最终失败；每episode至多一个。K1=BLOCKED_BY_NATURAL_EVENT_YIELD。

calibration的120条episode分解为：cream 53条成功、48条完整horizon失败但无anchor、19条合格；bowl分别51、58、11。两种cause标签没有参与事件筛选，因此缺口不来自额外label过滤。现有元数据不能把无anchor进一步分解成缺grasp、lift还是gripper，保留unknown。release-based与prerelease topology候选标签并列保存，后者不覆盖primary safe_success。详见MECHANISM_PHASE0B.md。

在诊断selection授权前，240条evaluation episode及其trace只做不解读内容的哈希/归属核验；yield由独立qualification元数据提供。不得把事后打开的内容回写成事前选样依据。

## Phase1完整网格与K2

| 任务 | 可用/计划事件 | core请求 | reference请求 | 全部请求 |
|---|---:|---:|---:|---:|
| cream | 19/20 | 10260 | 2052 | 12312 |
| bowl | 11/20 | 5940 | 1188 | 7128 |
| 合计 | 30/40 | 16200 | 3240 | 19440 |

bowl一个事件anchor316/320使部分prefix超出horizon，产生324 core+54 reference=378条真实结构性BLOCKED。全部请求原样保存；这些行不是safe_success=0观测。完整事件支持集排除该事件后，cream为19事件/16init cluster，bowl为10事件/10cluster。以下九档均使用该固定完整支持集，不能冒充计划20+20事件。

每(event,prefix,operator)先平均3个恢复seed，再对operator取oracle；事件内平均prefix、init内平均事件、任务内等权平均init。weakest为相同权重下最弱单臂。policy_call_cap固定8次恢复调用，另有1次generator validation。

| tail/action | cream oracle | cream weakest | cream gap | bowl oracle | bowl weakest | bowl gap |
|---|---:|---:|---:|---:|---:|---:|
| 4/8 | 2.0833 | 0.0000 | 2.0833 | 3.3333 | 0.0000 | 3.3333 |
| 4/16 | 17.0833 | 1.8750 | 15.2083 | 8.0000 | 0.0000 | 8.0000 |
| 4/32 | 52.5000 | 10.6250 | 41.8750 | 20.6667 | 10.6667 | 10.0000 |
| 8/8 | 2.0833 | 0.0000 | 2.0833 | 3.3333 | 0.0000 | 3.3333 |
| 8/16 | 19.1667 | 2.2917 | 16.8750 | 8.6667 | 0.0000 | 8.6667 |
| 8/32 | 56.4583 | 7.0833 | 49.3750 | 22.6667 | 10.0000 | 12.6667 |
| 16/8 | 2.0833 | 0.0000 | 2.0833 | 3.3333 | 0.0000 | 3.3333 |
| 16/16 | 21.0417 | 2.7083 | 18.3333 | 8.6667 | 0.0000 | 8.6667 |
| 16/32 | 56.6667 | 5.6250 | 51.0417 | 22.6667 | 9.3333 | 13.3333 |

表中oracle/weakest单位为%，gap单位为百分点。

两个任务没有共同满足oracle∈[25%,85%]且gap≥15pp的档位；bowl在全部九档都未满足。正式K2 receipt为BLOCKED且预算为null，同时保留样本不足与结构排除原因。独立诊断排序按min-task oracle、min-task gap降序，再action budget、tail horizon升序；排序首档为tail16/action32/cap8。须完成全量验收、提交并验证独立receipt后才能运行后续测量。

## 代码机理与配对效应

在tail16/action32、上述calibration完整支持集上，配对cluster估计为：

| 比较 | cream差值及95% CI | bowl差值及95% CI |
|---|---:|---:|
| hold − fresh_h4 | -25.6250pp [-36.4583,-15.2083] | +5.3333pp [2.0000,10.0000] |
| rollback − fresh_h16 | -44.7917pp [-62.7083,-27.9167] | -3.3333pp [-12.6667,5.3333] |
| fresh_h16 − fresh_h4 | -1.6667pp [-7.7083,4.1667] | -0.6667pp [-2.6667,1.3333] |

这些是calibration描述性结果，包含预算查看与多项比较，不作独立确认性“显著提升”主张。四臂绝对水平、九档配对效应、cluster值及10000次seed216214 bootstrap见operator_effects.json。

代码确定的变化是：hold增加一步六维零位移并保留夹爪；物体与接触动力学继续推进，并非冻结状态。rollback把cached prefix最后一动作old[k-1]的前六维取负并保留夹爪，这不是物理状态的逆操作。两者前导动作都占action budget，所以32步预算下只剩31步policy动作。fresh_h4/h16改变重规划间隔，实际执行间隔为min(命名horizon,configured tail)；tail4下1305对动作流与safe_success均一致，名称差异本身没有产生额外能力。

已核验hold、rollback与fresh_h16/fresh_h4三类双向六个案例。固定anchor/seed/chunk下，首动作或首次重新规划后轨迹分化；一个fresh_h4分支曾在control27的物理子步7–16短暂满足任务条件，但控制步末失去，而fresh_h16在control26子步10持续到步末，故实际branch得分不同。详见MECHANISM_PHASE1_CASES.md。完整trace支持状态、接触和成功时序的解释，但没有记录force/impulse，不推断未经测量的具体力学中介原因。所选预算的净升降案例另行补入最终报告。

### 选定诊断预算下的净效应配对轨迹核验

在tail16/action32/cap8下补充三对calibration案例，主线程独立重算六条gzip/content SHA、完整normalized contact、25 physics步/control步、checkpoint/D4、前缀动作及成功时点，全部PASS。可复现脚本为pai/verify_selected_mechanism_cases.py，完整证据为phase1/mechanism_selected_budget_parent_acceptance.json。每对同一event、generator/recovery seed、checkpoint及D4 detection/pre-tail；前两个动作完全相同，第一新动作index2才分叉。

| 配对 | 固定事件 / recovery seed | 观测结果 | 可观测机制链 |
|---|---|---|---|
| bowl hold vs fresh4 | init019 actor17 / seed29 | hold成功23个新动作；fresh4失败32步 | hold第一步为[0,0,0,0,0,0,1]，fresh4首先产生明显正X位移；后续release时点为18 vs29。hold于control24/physics6首次成功且control末仍成功；fresh4全程无成功，末端XY偏差0.04107m。|
| cream hold vs fresh4 | init018 actor29 / seed17 | hold失败32步；fresh4成功19步 | hold暂停笛卡尔动作但模拟器继续积分；fresh4首步向负Y/正Z运动。fresh4于control20/physics8成功且control末仍成功，gripper始终闭合；hold在22步release后失败，末端XY偏差0.03212m。|
| cream rollback vs fresh16 | init018 actor29 / seed7 | rollback失败32步；fresh16成功17步 | rollback首步前六维取上一动作负值，Z分量为-1，fresh16对应+0.81392；随后状态和接触轨迹分叉。rollback末端XY偏差0.06927m，fresh16于control18/physics9成功且control末仍成功。|

六条release_based_violation均为false，成败差异来自task_success。这些cream成功记录在gripper闭合时就满足环境谓词，不等于已经完成真实释放后稳定放置。hold/rollback的prelude计入新动作预算；六条任务剩余horizon均未裁剪，故这些配对不能归因于隐藏的任务horizon截断。第一分叉动作后的即时contact set仍相同，不能用“接触拓扑立刻改变”解释；差异体现为随后的状态、动作历史、release时点与接触演化。以上是选定匹配案例，不能据此确定摩擦、冲量、滑移等未测量中介，也不能外推为全部事件的普遍规律。

## Phase2：待独立selection授权后执行

当前尚未读取或运行evaluation，不能报告K3数值、两族成功率或crossing结论。

计划使用全部66个可用calibration+evaluation事件，四算子、八prefix、三恢复seed、四参考臂，共7128请求，其中2160个同配置calibration请求可复用Phase1原始证据，预计4968新请求。真实horizon越界继续单独保存。主统计split为evaluation；calibration仅描述性并列。U_A、U_B、U_full、max-k且None不补值、全部seed split-half划分、10000次(task,init) cluster bootstrap均保持冻结定义。

## 工程故障、资源与证据

| JobId | 工作 | 最终/当前状态 |
|---|---|---|
| dlcfesoi2j9tp8y5 | 旧cream采集 | Stopped，CPFS ESTALE，0正式episode |
| dlc189mgayv5nwjf | 旧bowl采集 | Stopped，全局停止，0正式episode |
| dlcwm6bd6bomjggq | 旧cream采集 | Stopped，cu130/driver不兼容，0正式episode |
| dlc1qayvbetb16ji | 旧bowl采集 | Stopped，同轮环境修复，0正式episode |
| dlc1hzmadm185c68 | 正式cream采集 | Succeeded，300/300 |
| dlc1rz7o5mf1vajn | 正式bowl采集 | Succeeded，300/300 |
| dlc14obea2cugtbf | cream grid r1 | Stopped，3896 COMPLETE及23取消 |
| dlc4vt6h6pcw6jzk | bowl grid r1 | Stopped，3133 COMPLETE、378结构越界、1超时、22取消 |
| dlcgtvdvu94wlto1 | cream grid r2 | Succeeded，12312/12312 |
| dlc1n05o22ewi5dx | bowl grid r2 | Succeeded，7128/7128 |

r1父进程先join再Queue.get，较大结果导致feeder阻塞；2MiB CPU反例复现，修复为先drain再join。46个失败/取消尝试和存在的11条trace在两job终态后逐SHA归档，7029 COMPLETE及378结构行保留，只重试相同请求。原超时请求重跑成功，结果pickle342396字节，新旧trace解压后15225576字节逐字节一致。只有IPC顺序改变，runtime/measurement/actor/seed/预算不变；兼容许可绑定唯一旧/新dispatch SHA。

统计复核发现未声明的bootstrap seed偏移，已在读取evaluation前统一为实际传入seed216214，10000次和统计定义未变；原PREREG/AMENDMENT没有被改写。诊断入口绑定真实Git object proof、协议SHA与calibration summary SHA，并重算选择结果。CPU回归与完整性回执按各自源提交保存，不重复累加测试数。

资源保持最多2job、每job2×A800/24CPU/200Gi，逐JobId核验robot quota下UseOversoldResource=true。截至Phase1完成保守累计9.064916 GPU小时，含失败启动、排队与dev14预留0.5 GPU小时；总上限20，控制器19.8提前停止。北京时间09:25/19:25起拒绝提交并停止本任务精确job，以覆盖09:30–09:40与19:30–19:40禁跑窗，:40后按持久分片恢复。

原始shard和trace逐文件SHA发布；冗余大JSONL同时保留完整无损gzip与带raw shard SHA索引的精简表，统计使用外部原始完整数据。SHA256SUMS覆盖所有发布产物；verify脚本PASS只证明证据完整性，不等于科学主张成功。

## 本阶段没有测的东西

没有训练actor，没有VLA、世界模型、RGB策略、扰动注入或真实机器人评测；不再主张性能优于baseline，不重新打开已冻结的universal hypothesis，不生成新idea，不启动S2。K1/K2失败、结构性不可执行行与低样本诊断结果均明确保留；不能把未满足原计划前提的测量包装为确认性结论。
