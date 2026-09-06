# Stage-2F / S1 预注册（Phase 0B 尚未启动）

## 冻结背景与主张

基线 main commit：`508e8a5c88780066a00d994c32d673d558594867`。
Stage-1b universal hypothesis=KILLED_IMMUTABLE；Stage-2B/2C/2D 原判决保持；Stage-2E-S0 产物 diagnostic_only。Stage-2D fresh-process 历史隔离 PASS 与最大重建误差 0.0 均保持，这不证明零注入可用。

本阶段唯一主张是恢复边界相对非嵌套恢复算子族的依赖是否非平凡，不再主张保留安全前缀能取得性能赢面。Phase 0A CPU 重算中，kLR 绝对 safe success（全 cohort、事件均值）从受限 38.10% 降到全支撑 19.44%；相对 immediate_fresh_h16 的全支撑 cluster Δ=+2.78 pp（95% CI [-5.56,+12.50] pp），replay-valid Δ=+1.59 pp（[-6.35,+11.90] pp），G0-1=INCONCLUSIVE；没有建立稳健性能优势，本阶段不再主张性能赢面。0A 只读取已公开冻结历史证据，不产生 S1 数据。

## 数据查看与执行状态

本文件登记所有 K 数值判据，必须在任何 Phase 0B rollout 前 commit；首次 commit 后不改写。当前尚无 S1 calibration/evaluation outcome、init pool 或 rollout。旧 Stage-2C 已发表 evaluation 表仅由计划明确要求的 Phase 0A 复现，不是 S1 evaluation。

GPU 执行前置阻塞：冻结 `fresh_process._child -> stage2d.runtime.execute_branch -> apply_perturbation` 无条件执行禁用路径。依附件硬约束 3，停止 GPU；未用 monkeypatch、零幅度、模块替换或重写被冻结 dispatcher 绕过。

正文要求不能因一个科学 gate 未达到而停止其他实验；执行口径为继续具备前置条件的独立工作，保留失败判决。缺少合格事件、无 K2 可选预算、零注入不满足时，不补数据、不擅自选预算、不把缺失结果判成 K3 阴性。本次在任何 K 判决之前即被代码契约阻塞。

## 固定设计

- 任务：put_the_cream_cheese_in_the_bowl、put_the_bowl_on_the_plate；排除 stove。
- actors 固定 shared multitask ACT seed 7/17/29，不训练 actor、不加 VLA/世界模型/RGB/扰动。
- 每任务 100 reset-randomized state；ID 0–9 infrastructure、10–49 calibration、50–89 evaluation、90–99 reserve。池生成属于获准 Phase 0B.1，当前未执行。
- Phase 0B：2 task × 3 seed × 100 init = 600 clean closed-loop rollouts，一个 rollout 至多一个结构事件。锚点：grasp、stable_lift_count≥2、closed gripper、task 未成功、remaining chunk=H_VALID=16。
- D1–D5 逐条沿用冻结 INSTRUMENTATION_S1.md；接触拓扑是规范排序完整 geom-name pair 集合与原始 pair，不以数量替代；release 标签与 release 前候选观测并列，均为离线纯函数。
- 每 branch 独立 spawn/env/RNG/cache，pid/env_hash/chunk_hash 缺任一即 BLOCKED，沿用全 isolation signature。
- Phase 1：tail_horizon={4,8,16} × action_budget={8,16,32}；每任务 20 calibration 事件 × 3 recovery actor seed × 4 operator × k={2,4,8,12,16}。完整 configured grid 为 21,600 branch，不含参考臂。policy-call cap 的可执行落定在 GPU 前置修正后、任何 S1 outcome 打开前单独预注册；当前不得执行未落定网格。
- U_A={fresh_h4,hold_1+fresh_h4}；U_B={fresh_h16,rollback_1+fresh_h16}；U_full=二者并集。按本任务明确的四算子定义执行，不根据 outcome 更换。
- Phase 2 在选定单一预算上，对全部 calibration/evaluation 事件 × 3 recovery actor seed × 4 operator × k={2,4,6,8,10,12,14,16}；immediate_fresh、fixed_delay_2/4/8 全程参考且不参与选择。
- R_U(k)=存在 u∈U，其三 recovery actor safe-success 均值≥2/3；来自冻结 stage2c/aggregate.py:67。k*=max{k:R_U(k)}；无定义为 None，来自 contracts.py:last_recoverable_prefix，不允许 fallback。
- 事件内先取 actor 均值，再按 (task,init_state_id) cluster bootstrap；10000 次，seed 216214；prefix row 不是独立样本。
- 所有预算、阈值、cause 候选阈值及选择仅可依 calibration；selection receipt 写入并提交后才允许打开 S1 evaluation 描述性 outcome。K1 所需 evaluation yield 计数必须由隔离的 qualification 管线提供，不能泄露恢复 outcome。

## K1 固定数值判据

两个任务各自 calibration 合格事件≥25 且 evaluation≥25；任一不足记 BLOCKED_BY_NATURAL_EVENT_YIELD。不能增加 init、重划 split 或把无事件伪造成失败 branch。当前 K1=NOT_EVALUATED。

## K2 固定数值判据

至少一个 configured budget 在两任务同时满足 oracle_best∈[0.25,0.85] 且 oracle_best−weakest_arm≥0.15。没有则 BLOCKED_BY_NO_WORKING_REGION，不擅选 Phase 2 预算。多档按原计划：oracle_best 降序、臂间差降序、action_budget 升序、tail_horizon 升序；多任务排序聚合与同分规则须在 GPU 前置修正后、任何 S1 数据前补充独立操作细则，不以观察到的结果选择。选定后写 selection_receipt.json 并 commit，不更改。当前 K2=NOT_EVALUATED。

## K3 固定数值判据

两任务必须各自同时满足：
1. 两族都有定义的事件数≥30。
2. minority_crossing_rate=min(mean[k_A>k_B],mean[k_B>k_A])≥0.15，且其 cluster-bootstrap 95% CI 下界严格高于族内 split-half 零分布 95 分位。
3. 两族均有定义事件上的 Spearman≤0.8。

噪声地板覆盖三个 recovery seed 全部非空 1-vs-2 划分（以及对换），同族按同样规则计算交叉率，报告零分布；None 缺失要并列报告，不能填充伪边界。两族各自平均 safe success 必须报告。未落定的小样本退化处理须在任何 S1 数据前单独封存操作细则。

全部满足才 NONTRIVIAL_OPERATOR_RELATIVITY，S2 只可另行启动；实测任一不满足才 MONOTONE_RESCALING_ONLY。本次前置阻塞，不得给这两个科学判决。当前 K3=NOT_EVALUATED。

## 资源、输出与停止

最多 2 PAI jobs、每 job≤2 A800，累计≤20 GPU 小时；CPU≤88 core/节点，memory≤用户当前指定 1.4 TB。资源意图为 robot 下闲时容量，实际 alias/GetJob/placement 尚未验证。每日 Asia/Shanghai 09:30 与 19:30 停止本工作流精确 JobIds，09:40 与 19:40 后才 resume；提交前必须先具备可验证的外部停机控制。当前 0 PAI jobs，0 GPU 小时。

所有新增实现和产物在 stage2f 下，另按明确产出要求新增 scripts/verify_r16p14_stage2f.py；冻结 a–e 路径不改。SHA256SUMS 覆盖全部 stage2f 产物、代码、报告及 verifier，自身不自哈希，Git commit 绑定该清单。不得伪造未执行网格的行文件。全部结束后不自行启动 S2。
