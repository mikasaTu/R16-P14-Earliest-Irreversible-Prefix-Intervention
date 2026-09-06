# Phase1 calibration mechanism cases

本报告是有限的、诊断性的个案机制分析，不估计总体成功率或处理效应，也不生成新 idea。输入严格限定为 Phase1 split=calibration、status=COMPLETE 的四个 core operator；没有打开 evaluation trace，也没有把个案当作总体效果。

## 扫描快照与配对规则

扫描时间（UTC）：2026-09-06T20:39:57.419106+00:00 — 2026-09-06T20:40:28.787806+00:00。扫描 7453 个 shard JSON；calibration COMPLETE 行 7029，四个 core operator 行 5852（fresh_h16=1465, fresh_h4=1463, hold_1+fresh_h4=1464, rollback_1+fresh_h16=1460），读取错误 0。

共 1454 个完整四臂组（put_the_bowl_on_the_plate=644, put_the_cream_cheese_in_the_bowl=810），ambiguous 0，不完整键 15。

分组键为 (task, event_id/event_instance_id, prefix_k, recovery_actor_seed, configured_tail_horizon, configured_action_budget, configured_policy_call_cap)；每组要求四个 core operator 各一条 COMPLETE calibration row。safe_success 不同才算 discordance。六对固定为三类比较的两个方向各一例，外加任务覆盖；只作为解释证据。

## 全扫描 discordance（不是效果估计）

| pair | 完整组分母 | discordant | A-up | B-up |
|---|---:|---:|---:|---:|
| hold_1+fresh_h4 vs fresh_h4 | 1454 | 139 | 49 | 90 |
| rollback_1+fresh_h16 vs fresh_h16 | 1454 | 189 | 14 | 175 |
| fresh_h16 vs fresh_h4 | 1454 | 61 | 35 | 26 |

A-up/B-up 表示该 pair 中 A/B 的 safe_success=true 而另一臂为 false；两方向均有观察，但不作方向性总体结论。

## 六对时序证据总表

首个对象 divergence 的括号是 A/B 在该 control step 的 object-target XY 距离；对象-目标接触首步、release control、task_success control 均为 A/B。actions 为新 recovery actions；calls 写成 recovery+validation。

| case | pair / safe A-B | 首个对象 divergence | 首个全 normalized topology divergence | 对象-目标接触首步 | release control | task_success control | new actions；calls |
|---|---|---|---|---|---|---|---|
| hold_up | hold_1+fresh_h4 vs fresh_h4 / True-False | c2 (0.1161/0.1135) | c19 | 22/19 | unknown/21 | 22/none | 21 ; 5+1 / 32 ; 8+1 |
| hold_down | hold_1+fresh_h4 vs fresh_h4 / False-True | c4 (0.1094/0.1060) | c18 | 18/21 | unknown/21 | none/21 | 32 ; 8+1 / 18 ; 5+1 |
| rollback_up | rollback_1+fresh_h16 vs fresh_h16 / True-False | c2 (0.0763/0.0665) | c28 | 28/unknown | unknown/32 | 28/none | 27 ; 4+1 / 32 ; 4+1 |
| rollback_down | rollback_1+fresh_h16 vs fresh_h16 / False-True | c2 (0.1214/0.1176) | c17 | 17/19 | 24/unknown | none/19 | 32 ; 4+1 / 18 ; 3+1 |
| fresh16_up | fresh_h16 vs fresh_h4 / True-False | c16 (0.0323/0.0322) | c23 | 26/27 | 20/21 | 26/27 | 15 ; 1+1 / 32 ; 8+1 |
| fresh16_down | fresh_h16 vs fresh_h4 / False-True | c20 (0.0331/0.0334) | c27 | 32/31 | 26/25 | none/31 | 32 ; 2+1 / 16 ; 4+1 |

## 个案细节

### hold_up: hold_1+fresh_h4 vs fresh_h4（A_up）

共同键：{'configured_action_budget': 32, 'configured_policy_call_cap': 8, 'configured_tail_horizon': 16, 'event_id': 'put_the_cream_cheese_in_the_bowl__init012__actor7', 'prefix_k': 2, 'recovery_actor_seed': 17, 'task': 'put_the_cream_cheese_in_the_bowl'}。safe_success A/B=True/False，task_success A/B=True/False。
anchor 对照：detection/pre_tail 相同=True/True；generator checkpoint、recovery checkpoint、chunk、env 相同=True/True/True/True；PID 不同=True。
动作/调用：A 21 new actions、5 recovery calls、1 validation、total 6；B 32、8、1、total 9。cap 8 只约束 recovery calls，total 可为 9。
首个 recovery action hash A/B=aeb42fef4924a9bd2abe763611882c97cf0928d30c70a77f6479778754e5a552/1123bd9f1b0145dc3a02b9bb0e9fb035ddedeec8d98f04a85591b520f890c7b2；physics/control 均为 {'25': 23}/{'25': 34}（每控制步 25）。
- 首个 object qpos divergence 在 control 2，XY 距离 A/B=0.1161/0.1135，max qpos diff=0.00526755。 首个 target qpos divergence 为 c19（max diff 0.00490629）。
- 首个全 normalized contact topology divergence 为 control 19；A-only=none；B-only=akita_black_bowl_1_g24/cream_cheese_1_g1, akita_black_bowl_1_g25/cream_cheese_1_g1, akita_black_bowl_1_g26/cream_cheese_1_g1。这说明接触集合何时分开，但不提供力或冲量因果证据。
- A hold_1+fresh_h4：对象-目标接触首 control 22；release control unknown、release XY=unknown、release violation=False；首次 task_success controls=22；末态 XY=0.0273、末态对象-目标 contacts=3、末态 task_success=True。
- B fresh_h4：对象-目标接触首 control 19；release control 21、release XY=0.0347、release violation=False；首次 task_success controls=none；末态 XY=0.0363、末态对象-目标 contacts=2、末态 task_success=False。
- contact_topology trace SHA：A 6853cecddbb6a1c7cab9c3ed282eaa0f759bc0ae1cc7bbb39df7c88a81d97618（字段匹配 True），B 55d97e65bb6c77274e31ab69d8cb87ae657a5cf5aabd7e48197bfb1b1dfc3712（字段匹配 True）。
- 机制边界：A 的 hold prelude 首个新动作是六维零位移、保留 gripper；B 直接使用 recovery actor 首动。轨迹显示首动后对象 qpos 分开，随后目标 qpos或接触集合在表中所列 control 分开；这支持调度差异对应轨迹差异，但不能单凭 trace 断言 hold 是成功或失败的充分原因。

### hold_down: hold_1+fresh_h4 vs fresh_h4（B_up）

共同键：{'configured_action_budget': 32, 'configured_policy_call_cap': 8, 'configured_tail_horizon': 8, 'event_id': 'put_the_cream_cheese_in_the_bowl__init018__actor29', 'prefix_k': 4, 'recovery_actor_seed': 29, 'task': 'put_the_cream_cheese_in_the_bowl'}。safe_success A/B=False/True，task_success A/B=False/True。
anchor 对照：detection/pre_tail 相同=True/True；generator checkpoint、recovery checkpoint、chunk、env 相同=True/True/True/True；PID 不同=True。
动作/调用：A 32 new actions、8 recovery calls、1 validation、total 9；B 18、5、1、total 6。cap 8 只约束 recovery calls，total 可为 9。
首个 recovery action hash A/B=aeb42fef4924a9bd2abe763611882c97cf0928d30c70a77f6479778754e5a552/cfc91a8c11e45044601cb522fa68c165f274adf456db720d4f603976fc6689ab；physics/control 均为 {'25': 36}/{'25': 22}（每控制步 25）。
- 首个 object qpos divergence 在 control 4，XY 距离 A/B=0.1094/0.1060，max qpos diff=0.00437677。 首个 target qpos divergence 为 c18（max diff 0.0232621）。
- 首个全 normalized contact topology divergence 为 control 18；A-only=akita_black_bowl_1_g10/table_collision, akita_black_bowl_1_g11/table_collision, akita_black_bowl_1_g12/table_collision, akita_black_bowl_1_g13/table_collision；B-only=none。这说明接触集合何时分开，但不提供力或冲量因果证据。
- A hold_1+fresh_h4：对象-目标接触首 control 18；release control unknown、release XY=unknown、release violation=False；首次 task_success controls=none；末态 XY=0.0668、末态对象-目标 contacts=0、末态 task_success=False。
- B fresh_h4：对象-目标接触首 control 21；release control 21、release XY=0.0209、release violation=False；首次 task_success controls=21；末态 XY=0.0209、末态对象-目标 contacts=2、末态 task_success=True。
- contact_topology trace SHA：A 4f8516b2b5186ceb8b054cac52d1c2419e2847d03796d21817a4c22d7b8b4e19（字段匹配 True），B f5a77c7ba424e63c1ec5c720068314bf5cf005d63086860c8ac7882119e22017（字段匹配 True）。
- 机制边界：A 的 hold prelude 首个新动作是六维零位移、保留 gripper；B 直接使用 recovery actor 首动。轨迹显示首动后对象 qpos 分开，随后目标 qpos或接触集合在表中所列 control 分开；这支持调度差异对应轨迹差异，但不能单凭 trace 断言 hold 是成功或失败的充分原因。

### rollback_up: rollback_1+fresh_h16 vs fresh_h16（A_up）

共同键：{'configured_action_budget': 32, 'configured_policy_call_cap': 8, 'configured_tail_horizon': 8, 'event_id': 'put_the_bowl_on_the_plate__init022__actor17', 'prefix_k': 2, 'recovery_actor_seed': 29, 'task': 'put_the_bowl_on_the_plate'}。safe_success A/B=True/False，task_success A/B=True/False。
anchor 对照：detection/pre_tail 相同=True/True；generator checkpoint、recovery checkpoint、chunk、env 相同=True/True/True/True；PID 不同=True。
动作/调用：A 27 new actions、4 recovery calls、1 validation、total 5；B 32、4、1、total 5。cap 8 只约束 recovery calls，total 可为 9。
首个 recovery action hash A/B=4d0424a62de830c42ecdbf021cb1acd84161431b5ff957360059c61d849e0daa/a0b0824335c11ec1f1236caf6af4ab1d649402b0971ec3f735ab2f0b86d01e7b；physics/control 均为 {'25': 29}/{'25': 34}（每控制步 25）。
- 首个 object qpos divergence 在 control 2，XY 距离 A/B=0.0763/0.0665，max qpos diff=0.0099239。 首个 target qpos divergence 为 c28（max diff 0.0121318）。
- 首个全 normalized contact topology divergence 为 control 28；A-only=akita_black_bowl_1_g38/plate_1_g1, akita_black_bowl_1_g40/plate_1_g10, akita_black_bowl_1_g40/plate_1_g2, akita_black_bowl_1_g40/plate_1_g3；B-only=akita_black_bowl_1_g30/gripper0_finger1_pad_collision, akita_black_bowl_1_g31/gripper0_finger1_pad_collision, plate_1_g5/table_collision, plate_1_g6/table_collision。这说明接触集合何时分开，但不提供力或冲量因果证据。
- A rollback_1+fresh_h16：对象-目标接触首 control 28；release control unknown、release XY=unknown、release violation=False；首次 task_success controls=28；末态 XY=0.0271、末态对象-目标 contacts=4、末态 task_success=True。
- B fresh_h16：对象-目标接触首 control unknown；release control 32、release XY=0.0215、release violation=False；首次 task_success controls=none；末态 XY=0.0211、末态对象-目标 contacts=0、末态 task_success=False。
- contact_topology trace SHA：A e6bcedc7a281f93c452698535054d095fafab6b536af5045c081de681f2a785e（字段匹配 True），B 022111dabdfc2132d44b5bd231534a5e7818c98d9f605c975188720350bf62d7（字段匹配 True）。
- 机制边界：A 的 rollback 首个新动作相对 control 1 的 cached prefix 末尾动作，前六维取负；trace 核对 max error=0.0，gripper 分量保留。源码实际使用 old[prefix_k-1]，这里是 prefix 末尾动作，不应笼统称 anchor 最后动作。随后对象 qpos、接触或目标关系按表分化；这仍是轨迹对应关系，不是动力学因果证明。

### rollback_down: rollback_1+fresh_h16 vs fresh_h16（B_up）

共同键：{'configured_action_budget': 32, 'configured_policy_call_cap': 8, 'configured_tail_horizon': 8, 'event_id': 'put_the_cream_cheese_in_the_bowl__init018__actor29', 'prefix_k': 2, 'recovery_actor_seed': 17, 'task': 'put_the_cream_cheese_in_the_bowl'}。safe_success A/B=False/True，task_success A/B=False/True。
anchor 对照：detection/pre_tail 相同=True/True；generator checkpoint、recovery checkpoint、chunk、env 相同=True/True/True/True；PID 不同=True。
动作/调用：A 32 new actions、4 recovery calls、1 validation、total 5；B 18、3、1、total 4。cap 8 只约束 recovery calls，total 可为 9。
首个 recovery action hash A/B=78632c7a269cc6d9a55f9ef251b906862ee62ff8b9ffe533694c97756033be1b/84a3bab83d22f53d51e6cfcdd1bfed00d55fa38b504362d9c54acd6bdee6c10e；physics/control 均为 {'25': 34}/{'25': 20}（每控制步 25）。
- 首个 object qpos divergence 在 control 2，XY 距离 A/B=0.1214/0.1176，max qpos diff=0.0113568。 首个 target qpos divergence 为 c17（max diff 0.000938417）。
- 首个全 normalized contact topology divergence 为 control 17；A-only=akita_black_bowl_1_g26/cream_cheese_1_g1, akita_black_bowl_1_g27/cream_cheese_1_g1；B-only=none。这说明接触集合何时分开，但不提供力或冲量因果证据。
- A rollback_1+fresh_h16：对象-目标接触首 control 17；release control 24、release XY=0.0545、release violation=False；首次 task_success controls=none；末态 XY=0.0688、末态对象-目标 contacts=3、末态 task_success=False。
- B fresh_h16：对象-目标接触首 control 19；release control unknown、release XY=unknown、release violation=False；首次 task_success controls=19；末态 XY=0.0242、末态对象-目标 contacts=2、末态 task_success=True。
- contact_topology trace SHA：A af0d7e39f5798921368c46c336926bcfb0abd993f91b81859851c5ec21880e33（字段匹配 True），B 442df854c42080ccd439eac8e89d5b23d0c34844e4fc095361ea48bda986f4e7（字段匹配 True）。
- 机制边界：A 的 rollback 首个新动作相对 control 1 的 cached prefix 末尾动作，前六维取负；trace 核对 max error=0.0，gripper 分量保留。源码实际使用 old[prefix_k-1]，这里是 prefix 末尾动作，不应笼统称 anchor 最后动作。随后对象 qpos、接触或目标关系按表分化；这仍是轨迹对应关系，不是动力学因果证明。

### fresh16_up: fresh_h16 vs fresh_h4（A_up）

共同键：{'configured_action_budget': 32, 'configured_policy_call_cap': 8, 'configured_tail_horizon': 16, 'event_id': 'put_the_cream_cheese_in_the_bowl__init018__actor29', 'prefix_k': 12, 'recovery_actor_seed': 7, 'task': 'put_the_cream_cheese_in_the_bowl'}。safe_success A/B=True/False，task_success A/B=True/False。
anchor 对照：detection/pre_tail 相同=True/True；generator checkpoint、recovery checkpoint、chunk、env 相同=True/True/True/True；PID 不同=True。
动作/调用：A 15 new actions、1 recovery calls、1 validation、total 2；B 32、8、1、total 9。cap 8 只约束 recovery calls，total 可为 9。
首个 recovery action hash A/B=321d7b22cb7dfeee4dac6603c74d87c1e1a6d465b899923d604483fb4011e6c1/321d7b22cb7dfeee4dac6603c74d87c1e1a6d465b899923d604483fb4011e6c1；physics/control 均为 {'25': 27}/{'25': 44}（每控制步 25）。
- 首个 object qpos divergence 在 control 16，XY 距离 A/B=0.0323/0.0322，max qpos diff=0.000200561。 首个 target qpos divergence 为 c26（max diff 0.0282636）。
- 首个全 normalized contact topology divergence 为 control 23；A-only=none；B-only=cream_cheese_1_g1/gripper0_finger2_collision。这说明接触集合何时分开，但不提供力或冲量因果证据。
- A fresh_h16：对象-目标接触首 control 26；release control 20、release XY=0.0243、release violation=False；首次 task_success controls=26；末态 XY=0.0261、末态对象-目标 contacts=2、末态 task_success=True。
- B fresh_h4：对象-目标接触首 control 27；release control 21、release XY=0.0254、release violation=False；首次 task_success controls=27；末态 XY=0.0480、末态对象-目标 contacts=3、末态 task_success=False。
- contact_topology trace SHA：A cb1af4bbd9fc7c144defaca088a8ca0851d45736ebf4ce6736cc2361b2e2a9aa（字段匹配 True），B 0d732000b725447e5e23c4d6dd2dd76e4d463f382152cf1cc5e6903dd98eb700（字段匹配 True）。
- 机制边界：A/B 首个 recovery action hash 相同；首次对象 qpos 分化晚于 prefix，符合前若干 actor actions 相同、随后 effective horizon 16 与 4 导致 chunk 重规划时点不同的证据。其后 contact、target、release、success 时序不同，但没有力记录，具体动力学原因 unknown。

### fresh16_down: fresh_h16 vs fresh_h4（B_up）

共同键：{'configured_action_budget': 32, 'configured_policy_call_cap': 8, 'configured_tail_horizon': 16, 'event_id': 'put_the_cream_cheese_in_the_bowl__init020__actor17', 'prefix_k': 16, 'recovery_actor_seed': 7, 'task': 'put_the_cream_cheese_in_the_bowl'}。safe_success A/B=False/True，task_success A/B=False/True。
anchor 对照：detection/pre_tail 相同=True/True；generator checkpoint、recovery checkpoint、chunk、env 相同=True/True/True/True；PID 不同=True。
动作/调用：A 32 new actions、2 recovery calls、1 validation、total 3；B 16、4、1、total 5。cap 8 只约束 recovery calls，total 可为 9。
首个 recovery action hash A/B=c5f35aa594b263567ff5ad3b518c3b779e60174a6c2c943782a4e682917638c5/c5f35aa594b263567ff5ad3b518c3b779e60174a6c2c943782a4e682917638c5；physics/control 均为 {'25': 48}/{'25': 32}（每控制步 25）。
- 首个 object qpos divergence 在 control 20，XY 距离 A/B=0.0331/0.0334，max qpos diff=0.000613039。 首个 target qpos divergence 为 c31（max diff 0.0233193）。
- 首个全 normalized contact topology divergence 为 control 27；A-only=cream_cheese_1_g1/gripper0_finger1_collision, cream_cheese_1_g1/gripper0_finger2_collision；B-only=none。这说明接触集合何时分开，但不提供力或冲量因果证据。
- A fresh_h16：对象-目标接触首 control 32；release control 26、release XY=0.0269、release violation=False；首次 task_success controls=none；末态 XY=0.0514、末态对象-目标 contacts=3、末态 task_success=False。
- B fresh_h4：对象-目标接触首 control 31；release control 25、release XY=0.0278、release violation=False；首次 task_success controls=31；末态 XY=0.0274、末态对象-目标 contacts=3、末态 task_success=True。
- contact_topology trace SHA：A f3d4403df120513c427fd1ef25b5234889ac6504507af1797869749745d1d3a3（字段匹配 True），B 749eeb972f1dfd656c69cec84e8ac73eba48f8dd2bd3f06db8376538f7dcd80f（字段匹配 True）。
- 机制边界：A/B 首个 recovery action hash 相同；首次对象 qpos 分化晚于 prefix，符合前若干 actor actions 相同、随后 effective horizon 16 与 4 导致 chunk 重规划时点不同的证据。其后 contact、target、release、success 时序不同，但没有力记录，具体动力学原因 unknown。

## labels、接触与因果边界

六对入选臂的 release-based violation 都为 false；safe_success 差别由 task_success 取值造成。对象-目标接触也不是充分条件：例如 rollback_down 的失败臂后来出现对象-目标接触并 release，末态仍 task_success=false。release index 记录 transition 时刻，但不能从 XY 距离或接触集合单独推出任务判定原因。

本报告只确证：固定 anchor 对照下，hold/rollback 首动或 fresh16/fresh4 批长改变 action stream；trace 中对象/目标 qpos、normalized contact topology、release transition 与 task_success 的时序随后不同。trace 没有力、冲量或反事实控制，因此更具体的动力学因果解释均为 unknown，不把个案方向外推为总体效果。

## 源码对应的可证机制

- experiments/r16_p14_stage2f/s1/runtime.py:60-63 固定四个 operator 与 prelude；:66-117 计算 effective horizon 与独立的新动作预算。
- runtime.py:375-381 replay old[:prefix_k] 并在 prefix 后生成 pre_tail signature；:389-407 执行最多一个 prelude，再受 recovery cap、effective horizon 和 action budget 调度。
- runtime.py:391 的 rollback 参数明确是 old[int(prefix_k)-1]，即 cached prefix 末尾动作；experiments/r16_p14_stage2c/r16_p14_stage2c/runtime.py:392-400 中 rollback 前六维取负、hold 前六维为零且保留 gripper。
- runtime.py:409-419 写 trace 与 provenance；:411-413 定义 safe_success = task_success and not release_based_violation；:443-449 明确 recovery calls、单独 validation call 和 total call accounting。

## 复核身份

源码树 commit（扫描时记录）：dc482d85b82478fbb8ed5452e4df9fa96956d06d。源码 SHA256：
- stage2c_runtime.py: bc25cb9f3ad15ac627d24b02d6b3de982f0e48044fbaeb7b71d57c7d37eee051 (17543 bytes)
- stage2f_measurement.py: b31dc34270905c6d76ea342c6a1deef8739c53e8bc9813e38d127ee48aab8c62 (28919 bytes)
- stage2f_s1_runtime.py: 3203db2452eef4bf6fe8b304eae2b14b1dc7ae80630479317c4610f063cf4417 (25881 bytes)

机器可读的完整计数、6 对 row metadata、12 条 trace SHA 与时序 divergence 字段见 artifacts/stage2f/phase1/mechanism_cases.json。
