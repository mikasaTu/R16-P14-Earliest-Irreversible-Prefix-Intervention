# S1 执行细则补充（任何 S1 rollout 之前封存）

本补充不修改 PREREG_S1.md 中 K1/K2/K3 数值。用户继续完成全部实验的指令授权在 Stage-2F 新建零注入 spawn dispatcher；Stage-2D fresh_process/isolation 仍直接 import 用作冻结隔离合同参照，但禁用它调用无条件注入的分支入口。冻结 2C/2D 文件不改动；重建仅调用 D.reconstruct_anchor，hold/rollback 复用 C 纯动作函数。没有调用 perturbation/target shift/path blocker 的路径。

## 事先固定的执行定义
- 每任务 init id 0..99 对应 reset seed=2016214+10000*task_index+init_id，任务顺序 cream_cheese_in_bowl、bowl_on_plate。只接受此 100 个状态，非有限值、初始成功或重复 hash 直接记基础设施错误，不补样、不换 seed。固定 split 沿预注册。
- clean actor 每控制步以四状态三动作历史预测 16 步 chunk，执行首动作；结构锚点为动作前第一个满足 grasp、stable_lift_count>=2、上一动作 gripper>0、尚未成功且预测长度16的状态。继续完整 clean episode 到成功或固定任务 horizon。合格自然失败事件=有结构锚点且 clean episode 最终未成功；每 episode 至多1个。不使用恢复 outcome 选择事件。
- K1 evaluation 只开放合格事件计数。evaluation 事件及 clean outcome 密封存储；Phase 1 只读取 calibration。选择 receipt Git commit 之前不运行/读取 evaluation 恢复分支。
- 预算固定 policy_call_cap=8；每次重新预测实际执行 min(算子命名 horizon, configured tail_horizon, remaining action_budget) 步。action_budget 包含 hold/rollback 前导动作及 fresh 动作，不包含缓存前缀 k。环境到达任务固定 horizon 后停止，绝不将实际用量替代 configured 档位。
- 参考 immediate_fresh 固定检测点 k=2，fixed_delay_d 为 k=2+d，随后 fresh_h16；同 configured 预算。参考臂不参与 K2、族或边界选择。
- Phase 1 calibration 事件按 (init_state_id,generator_actor_seed,event_instance_id) 排序取前20；不足20不补。若 K1 未通过但有足够 calibration，依用户正文继续独立 Phase 1，并保留 K1 失败；Phase 2 仍需真实 K2 可选预算，不创造预算。
- grid oracle：每(event,k,operator) 先平均3个恢复seed，再对operator取最大；事件内平均prefix、init内平均事件、任务内等权平均init。weakest_arm 为同一权重下最弱的单个operator；gap=oracle-weakest。报告每 prefix 及 completeness。
- 多任务档位排序：min_task(oracle) 降序、min_task(gap) 降序、action_budget 升序、tail_horizon 升序。仅所有 task 都通过 K2 的档位可选；选择 receipt 提交后锁定。
- boundary R_U 使用>=2/3；split-half 的singleton和另2seed各自以该阈值判定。覆盖3个 singleton-vs-complement 划分，两族分别计算。交换两半的 minority crossing 相同，保留对应映射。null 为各划分的 task/init cluster bootstrap minority crossing 的合并分布，95分位作该任务noise floor。bootstrap=10000，seed216214；绝不把prefix视为独立样本。
- crossing主分析只用两族都定义的事件，并列报告 None 数及分母。常量边界 Spearman=None，K3不通过；缺失/错误分支记BLOCKED而不伪造阴性。K3最低事件数30保持原值。
- D1 每物理 simulation step 记录 raw geom pairs 以及去重规范排序集合，并有control_step/substep键。D2 release-based几何阈值用冻结 TaskSpec 目标容差/抬升阈值和原始目标；新增非目标接触拓扑相对anchor基线，只在已抬升且未release时检查，排除机器人/目标geom。两种标签分别输出，primary safe_success 用既有 release label，候选量不覆盖primary。缺失信息显式报错/缺失。
- 所有训练/推理共用最多2个job、每job2GPU、20 GPU小时预算。新事件/恢复按原子持久分片恢复，完成片不可覆盖。资源只接受 robot quota 下 UseOversoldResource=true 的闲时实例；不满足即停止，不占专属。
- 外部控制每日北京时间09:30/19:30前先停止本run精确job id，并验证Stopped；09:40/19:40后才恢复。预算/黑窗同时由payload拒绝新branch和外部控制检查。该控制验证前不提交PAI。
