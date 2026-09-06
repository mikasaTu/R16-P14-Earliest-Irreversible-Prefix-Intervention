# Phase1 calibration mechanism cases

本报告是有限的、诊断性的个案机制分析；不估计总体成功率、处理效应或生成新 idea。所有输入行限于 Phase1 `split=calibration` 且 `status=COMPLETE`。evaluation 行和 evaluation trace 没有进入分析。

## 扫描快照与配对规则

扫描时间（UTC）：`2026-09-06T20:28:34.237479+00:00` — `2026-09-06T20:29:04.353311+00:00`。扫描了 `6831` 个 shard JSON；calibration COMPLETE 行总数为 `6453`，四个 core operator 行为 `5372`（fresh_h16=1345, fresh_h4=1343, hold_1+fresh_h4=1344, rollback_1+fresh_h16=1340），读取错误 `0`。

每个四臂组的精确键为 `(task, event_id/event_instance_id, prefix_k, recovery_actor_seed, configured_tail_horizon, configured_action_budget, configured_policy_call_cap)`；四个 operator 必须各有且仅有一条 COMPLETE calibration core row，reference operator 不参与。得到 `1335` 个完整组（put_the_bowl_on_the_plate=585, put_the_cream_cheese_in_the_bowl=750），ambiguous 全臂组 `0`，不完整键 `14`。

候选先按 `safe_success` 不同筛选，再按方向和 operator pair 预先选取最多四对；这些对用于解释 discordance，不是按成功率排名，也不是总体效果样本。

## 全扫描 discordance（不是效果估计）

| pair | 完整组分母 | discordant | A-up | B-up |
|---|---:|---:|---:|---:|
| hold_1+fresh_h4 vs fresh_h4 | 1335 | 128 | 48 | 80 |
| rollback_1+fresh_h16 vs fresh_h16 | 1335 | 177 | 13 | 164 |
| fresh_h16 vs fresh_h4 | 1335 | 48 | 30 | 18 |

A/B-up 只表示该 pair 中 A 或 B 的 `safe_success=true` 而另一臂为 false。三个 pair 的两个方向均有观测；下面只深入四对。

## 入选配对证据

### hold_up: hold_1+fresh_h4 vs fresh_h4（A_up）

共同键：`{'task': 'put_the_cream_cheese_in_the_bowl', 'event_id': 'put_the_cream_cheese_in_the_bowl__init012__actor7', 'prefix_k': 2, 'recovery_actor_seed': 17, 'configured_tail_horizon': 16, 'configured_action_budget': 32, 'configured_policy_call_cap': 8}`。A=`hold_1+fresh_h4`，B=`fresh_h4`；safe_success A/B=`True/False`，task_success A/B=`True/False`。

- anchor/detection/pre_tail：detection 相同=`True`，pre_tail 相同=`True`；generator checkpoint、recovery checkpoint、original chunk、env hash 分别相同=`True/True/True/True`；PID 不同=`True`。
- trace：A `598` records（`{'physics_step': 575, 'action_step': 23}`），B `884` records（`{'physics_step': 850, 'action_step': 34}`）；两臂每控制步均为 25 physics steps（A `{'25': 23}`，B `{'25': 34}`）。trace 字段与现场 SHA 均匹配=`[True, True]`。
- 首个缓存动作 hash A/B=`94bbf7734e3e8055ebd58094717b56edd87c63c7c3f5e57bd32986cb4c5b249e/94bbf7734e3e8055ebd58094717b56edd87c63c7c3f5e57bd32986cb4c5b249e`；首个 recovery action 在 control step A/B=`2/2`，hash A/B=`aeb42fef4924a9bd2abe763611882c97cf0928d30c70a77f6479778754e5a552/1123bd9f1b0145dc3a02b9bb0e9fb035ddedeec8d98f04a85591b520f890c7b2`。
- 新动作/调用：A `21` actions，recovery calls `5`，validation `1`，total `6`；B `32` actions，recovery calls `8`，validation `1`，total `9`。`policy_call_cap=8` 只约束 recovery calls；另有 1 次 generator validation，所以 total 可为 9。
- release label A/B：violation=`False/False`，release indices=`[]/[571]`；selected topology candidate=`False/False`。因此本对的 safe_success 差别来自 task_success 的观测差异，而不是 release-based violation。
- trace SHA：A `6853cecddbb6a1c7cab9c3ed282eaa0f759bc0ae1cc7bbb39df7c88a81d97618`；B `55d97e65bb6c77274e31ab69d8cb87ae657a5cf5aabd7e48197bfb1b1dfc3712`。路径分别为 `/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r16_p14_stage2f/s1-20260907/artifacts/stage2f/phase1/contact_topology/put_the_cream_cheese_in_the_bowl/ebde14d988a312e6df297c0a37022b0f391eb32095bcc786d4679e91018814fa.jsonl.gz` 和 `/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r16_p14_stage2f/s1-20260907/artifacts/stage2f/phase1/contact_topology/put_the_cream_cheese_in_the_bowl/fe4656d8f775fa34c663d2a4b6e15a1e02624a725dbcc800a311e44b84d2e508.jsonl.gz`。

证据范围内：hold prelude 的首个 recovery action 是全零位移、保留 gripper=1 的 hold action；同锚点 fresh_h4 首个 recovery action 是 recovery actor 输出。hold 臂在 21 个新动作后成功，fresh_h4 在 32 个新动作/8 recovery calls 后仍失败。

### hold_down: hold_1+fresh_h4 vs fresh_h4（B_up）

共同键：`{'task': 'put_the_cream_cheese_in_the_bowl', 'event_id': 'put_the_cream_cheese_in_the_bowl__init018__actor29', 'prefix_k': 4, 'recovery_actor_seed': 29, 'configured_tail_horizon': 8, 'configured_action_budget': 32, 'configured_policy_call_cap': 8}`。A=`hold_1+fresh_h4`，B=`fresh_h4`；safe_success A/B=`False/True`，task_success A/B=`False/True`。

- anchor/detection/pre_tail：detection 相同=`True`，pre_tail 相同=`True`；generator checkpoint、recovery checkpoint、original chunk、env hash 分别相同=`True/True/True/True`；PID 不同=`True`。
- trace：A `936` records（`{'physics_step': 900, 'action_step': 36}`），B `572` records（`{'physics_step': 550, 'action_step': 22}`）；两臂每控制步均为 25 physics steps（A `{'25': 36}`，B `{'25': 22}`）。trace 字段与现场 SHA 均匹配=`[True, True]`。
- 首个缓存动作 hash A/B=`2a9d8c6c63474b40aca53977684e5c499f2e74d9a009b6e46de2b8eec3797478/2a9d8c6c63474b40aca53977684e5c499f2e74d9a009b6e46de2b8eec3797478`；首个 recovery action 在 control step A/B=`4/4`，hash A/B=`aeb42fef4924a9bd2abe763611882c97cf0928d30c70a77f6479778754e5a552/cfc91a8c11e45044601cb522fa68c165f274adf456db720d4f603976fc6689ab`。
- 新动作/调用：A `32` actions，recovery calls `8`，validation `1`，total `9`；B `18` actions，recovery calls `5`，validation `1`，total `6`。`policy_call_cap=8` 只约束 recovery calls；另有 1 次 generator validation，所以 total 可为 9。
- release label A/B：violation=`False/False`，release indices=`[]/[571]`；selected topology candidate=`False/False`。因此本对的 safe_success 差别来自 task_success 的观测差异，而不是 release-based violation。
- trace SHA：A `4f8516b2b5186ceb8b054cac52d1c2419e2847d03796d21817a4c22d7b8b4e19`；B `f5a77c7ba424e63c1ec5c720068314bf5cf005d63086860c8ac7882119e22017`。路径分别为 `/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r16_p14_stage2f/s1-20260907/artifacts/stage2f/phase1/contact_topology/put_the_cream_cheese_in_the_bowl/cf43f6eaf4e72998e5daebc544a4eba49d1a0e2f3883386df96d13724bd70d3a.jsonl.gz` 和 `/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r16_p14_stage2f/s1-20260907/artifacts/stage2f/phase1/contact_topology/put_the_cream_cheese_in_the_bowl/0790d63c47585cf1e2b37255a992bf5cc787dfa0c00cd721ad62de3a0df09fd5.jsonl.gz`。

证据范围内：hold prelude 仍先插入 hold action；该臂 32 actions 后失败，而 fresh_h4 用 actor 首动并在 18 actions 后成功。两者 release label 都无 violation。

### rollback_up: rollback_1+fresh_h16 vs fresh_h16（A_up）

共同键：`{'task': 'put_the_bowl_on_the_plate', 'event_id': 'put_the_bowl_on_the_plate__init022__actor17', 'prefix_k': 2, 'recovery_actor_seed': 29, 'configured_tail_horizon': 8, 'configured_action_budget': 32, 'configured_policy_call_cap': 8}`。A=`rollback_1+fresh_h16`，B=`fresh_h16`；safe_success A/B=`True/False`，task_success A/B=`True/False`。

- anchor/detection/pre_tail：detection 相同=`True`，pre_tail 相同=`True`；generator checkpoint、recovery checkpoint、original chunk、env hash 分别相同=`True/True/True/True`；PID 不同=`True`。
- trace：A `754` records（`{'physics_step': 725, 'action_step': 29}`），B `884` records（`{'physics_step': 850, 'action_step': 34}`）；两臂每控制步均为 25 physics steps（A `{'25': 29}`，B `{'25': 34}`）。trace 字段与现场 SHA 均匹配=`[True, True]`。
- 首个缓存动作 hash A/B=`025af765306f6fb0734033cbc84097a16542cd83d9dda0c00b1bfb93db03fd22/025af765306f6fb0734033cbc84097a16542cd83d9dda0c00b1bfb93db03fd22`；首个 recovery action 在 control step A/B=`2/2`，hash A/B=`4d0424a62de830c42ecdbf021cb1acd84161431b5ff957360059c61d849e0daa/a0b0824335c11ec1f1236caf6af4ab1d649402b0971ec3f735ab2f0b86d01e7b`。
- 新动作/调用：A `27` actions，recovery calls `4`，validation `1`，total `5`；B `32` actions，recovery calls `4`，validation `1`，total `5`。`policy_call_cap=8` 只约束 recovery calls；另有 1 次 generator validation，所以 total 可为 9。
- release label A/B：violation=`False/False`，release indices=`[]/[857]`；selected topology candidate=`False/False`。因此本对的 safe_success 差别来自 task_success 的观测差异，而不是 release-based violation。
- trace SHA：A `e6bcedc7a281f93c452698535054d095fafab6b536af5045c081de681f2a785e`；B `022111dabdfc2132d44b5bd231534a5e7818c98d9f605c975188720350bf62d7`。路径分别为 `/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r16_p14_stage2f/s1-20260907/artifacts/stage2f/phase1/contact_topology/put_the_bowl_on_the_plate/8e21da14d1926376a04aa7ad7b8fff388100e511a8aad3d8532d59257438d736.jsonl.gz` 和 `/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r16_p14_stage2f/s1-20260907/artifacts/stage2f/phase1/contact_topology/put_the_bowl_on_the_plate/b22cfd7ad32f240845dc86238ed8c033c1d6f460f42335fc9a2d46b6424097f6.jsonl.gz`。

证据范围内：rollback 臂首个 recovery action 与 fresh_h16 不同；rollback action 的前六维是 anchor 最后动作的取负（D2 定义）。rollback 臂 27 actions 后成功，fresh_h16 32 actions 后失败。

### fresh16_up: fresh_h16 vs fresh_h4（A_up）

共同键：`{'task': 'put_the_cream_cheese_in_the_bowl', 'event_id': 'put_the_cream_cheese_in_the_bowl__init018__actor29', 'prefix_k': 12, 'recovery_actor_seed': 7, 'configured_tail_horizon': 16, 'configured_action_budget': 32, 'configured_policy_call_cap': 8}`。A=`fresh_h16`，B=`fresh_h4`；safe_success A/B=`True/False`，task_success A/B=`True/False`。

- anchor/detection/pre_tail：detection 相同=`True`，pre_tail 相同=`True`；generator checkpoint、recovery checkpoint、original chunk、env hash 分别相同=`True/True/True/True`；PID 不同=`True`。
- trace：A `702` records（`{'physics_step': 675, 'action_step': 27}`），B `1144` records（`{'physics_step': 1100, 'action_step': 44}`）；两臂每控制步均为 25 physics steps（A `{'25': 27}`，B `{'25': 44}`）。trace 字段与现场 SHA 均匹配=`[True, True]`。
- 首个缓存动作 hash A/B=`2a9d8c6c63474b40aca53977684e5c499f2e74d9a009b6e46de2b8eec3797478/2a9d8c6c63474b40aca53977684e5c499f2e74d9a009b6e46de2b8eec3797478`；首个 recovery action 在 control step A/B=`12/12`，hash A/B=`321d7b22cb7dfeee4dac6603c74d87c1e1a6d465b899923d604483fb4011e6c1/321d7b22cb7dfeee4dac6603c74d87c1e1a6d465b899923d604483fb4011e6c1`。
- 新动作/调用：A `15` actions，recovery calls `1`，validation `1`，total `2`；B `32` actions，recovery calls `8`，validation `1`，total `9`。`policy_call_cap=8` 只约束 recovery calls；另有 1 次 generator validation，所以 total 可为 9。
- release label A/B：violation=`False/False`，release indices=`[545]/[571]`；selected topology candidate=`False/False`。因此本对的 safe_success 差别来自 task_success 的观测差异，而不是 release-based violation。
- trace SHA：A `cb1af4bbd9fc7c144defaca088a8ca0851d45736ebf4ce6736cc2361b2e2a9aa`；B `0d732000b725447e5e23c4d6dd2dd76e4d463f382152cf1cc5e6903dd98eb700`。路径分别为 `/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r16_p14_stage2f/s1-20260907/artifacts/stage2f/phase1/contact_topology/put_the_cream_cheese_in_the_bowl/493442dcdf63e210baf2da0916cb9b6f3e42d5bb41f2aae424d95e6d0384902d.jsonl.gz` 和 `/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r16_p14_stage2f/s1-20260907/artifacts/stage2f/phase1/contact_topology/put_the_cream_cheese_in_the_bowl/b3fff980cf13a5da8b658eac59cd44cc94360517f8e3522cb796ff3539d0d6c5.jsonl.gz`。

证据范围内：fresh_h16 与 fresh_h4 的首个 recovery action hash 相同；差别首先是 effective horizon 16 对 4，随后新动作数为 15 对 32、recovery calls 为 1 对 8，成功臂提前完成。

## 源码对应的可证机制

- `s1/runtime.py:60-63` 固定四个 operator 与 hold/rollback prelude；`stage2c/runtime.py:392-400` 给出 rollback 前六维取负、hold 只保留 gripper 的具体动作。
- `s1/runtime.py:66-117` 计算 `effective_execution_horizon=min(operator_horizon, configured_tail_horizon)` 并把新动作预算与 cached prefix 分开；这解释了 fresh16/fresh4 的同首动、不同后续批长。
- `s1/runtime.py:375-381` 先 replay `old[:prefix_k]`，再生成 pre_tail signature；本报告配对中 detection/pre_tail 完整 signature hash 相同，支持共同 anchor 对照。
- `s1/runtime.py:389-407` 先执行最多一个 hold/rollback prelude，再以 recovery call 取 chunk、按 effective horizon 执行动作并受 action budget/cap 限制；`443-449` 明确 `recovery_policy_calls`、`validation_policy_calls=1` 与 `total_policy_calls=1+recovery`。
- `s1/runtime.py:409-419` 关闭并标注 trace；`411-413` 定义 `safe_success = task_success and not release_based_violation`。四个个案的 release label 都没有 violation，故不能把差别说成 release-outside-target 机制。

这些行与 trace 只确证调度顺序、首动、批长、调用数和观测结果；它们支持‘operator 改变了 recovery action stream，随后 task_success 不同’这一范围内推断，不能单凭个案证明动力学因果或总体收益。

## 复核身份

源码树 commit（扫描时记录）：`dc482d85b82478fbb8ed5452e4df9fa96956d06d`。源码 SHA256：
- `stage2f_s1_runtime.py` `3203db2452eef4bf6fe8b304eae2b14b1dc7ae80630479317c4610f063cf4417` (25881 bytes)
- `stage2c_runtime.py` `bc25cb9f3ad15ac627d24b02d6b3de982f0e48044fbaeb7b71d57c7d37eee051` (17543 bytes)
- `stage2f_measurement.py` `b31dc34270905c6d76ea342c6a1deef8739c53e8bc9813e38d127ee48aab8c62` (28919 bytes)

机器可读完整计数、四对 row metadata 和每条 contact_topology trace SHA 见 `artifacts/stage2f/phase1/mechanism_cases.json`。
