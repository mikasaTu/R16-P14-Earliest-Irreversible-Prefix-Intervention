# R16-P14 Stage-2B Action-Chunk Faithfulness and Actor-Conditioned Replanability Pilot

## 结论摘要

最终枚举为 **`BLOCKED_BY_ACTOR_CONDITIONED_PERTURBATION`**。本轮按用户最新指令完成 Phase A–F；即使 A–D gate 失败也继续执行后续矩阵，但 gate 失败后的 Atlas/Operator 结果只作描述，不获得正标签。`accepted=false`。

- Action-chunk executability: **PASS**, `H_valid=16`（要求 >=8）。
- Actor-conditioned perturbation: **BLOCKED**。
- Full actor-history replay: **BLOCKED**。
- Track A conditional timing: **INCONCLUSIVE**。
- Track B oracle replanability: **INCONCLUSIVE**。
- Local repair: **NO_SIGNAL**；operator router: **NO_OPERATOR_ROUTING_SIGNAL**。

## 不可变负结论与证据冻结

Stage-1b universal hypothesis 仍为 **`KILLED_IMMUTABLE`**；本轮不能删除、弱化或反转它。Stage-2A checkpoint 仍为 **`PASS_READY_FOR_BOUNDED_ATLAS`**，当时 Track A/B 均为 INCONCLUSIVE，且未运行 full atlas。

- Stage-1 actual resolvable commit/tree: `ee56aa096e308214c38132d0e6d2a9e576c29792` / `50e56b84c5e867447b22f81e31454646a12a9eb8`。
- Stage-1b commit/tree: `e29e3ead42fd1799b412a4968e6a67aac3784874` / `4016b05942e6dfb291f1bc3a2644e177a208b608`。
- Stage-2A raw evidence commit/tree: `df20e31cefc4db22ba23f2b61284469e781d5558` / `bfe056abce8550cfc3ece69cfd3198f690728dfa`。
- Stage-2A report commit/tree: `dd86afbb445169ecef6eb25fb0a73d09763585b5` / `e69f4e468ee64e2ab7ceaed330c0dad32265e89f`。
- Step4 pre-report evidence HEAD/tree: `13bc2d13bc8728f2a19cd8cac8a6bb70d9c56f0e` / `dd846bb89b51673af9d732beee5d90ea8e597649`。
- Stage-2A manifest 中旧的 Stage-1 引用 `a1b611...` 在当前仓库不可解析；本轮未掩盖该事实，冻结文件将它标为 historical-unresolvable，并采用当前可验证发布对象 `ee56aa...`。

## Phase A — Action-chunk executability

共运行 `600` 个 clean episodes；h=1 aggregate success=`0.667`，late reach=`0.708`。

| h | success | late reach | prefix faithful | late anchors | errors | pass |
|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 0.667 | 0.708 | 0.988 | 84 | 0 | yes |
| 2 | 0.850 | 0.925 | 0.991 | 111 | 0 | yes |
| 4 | 0.842 | 0.892 | 1.000 | 107 | 0 | yes |
| 8 | 0.883 | 0.933 | 1.000 | 112 | 0 | yes |
| 16 | 0.933 | 0.950 | 1.000 | 98 | 0 | yes |

Phase A 使用 target-aware object-drop monitor v2。这是一项透明的 post-registration measurement repair：首轮完整 aggregate 将 stove 的正确 placement descent 误标为 drop；旧证据已按 SHA256 隔离，修复发生在 Phase B 和任何扰动/replan outcome 之前。详见 `reports/phase_a_monitor_bug_audit.md`。

若该 gate 被阻断，它只说明冻结 ACT substrate 不适于 prefix 实验，不解释为 R16-P14 mechanism failure。

## Phase B — Actor-generated events

从 `180` 个 ACT h=1 closed-loop attempts 构建 `129` 个 eligible events；formal nominal 全部来自 actor chunk，demonstration nominal count=`0`。Reserved IDs 30–49 未 rollout/未查看 outcome。

| task | eligible total | qualification | calibration | evaluation |
|---|---:|---:|---:|---:|
| put_the_cream_cheese_in_the_bowl | 59 | 18 | 20 | 21 |
| put_the_bowl_on_the_stove | 70 | 23 | 24 | 23 |

## Phase C — Actor-conditioned perturbation qualification

选择只读取 phase validity、immediate/delayed cause、first offset 与 replay；没有读取 immediate-replan、last-safe、k_best 或 method gain。

| task | tested severity | valid | errors | immediate | delayed | median offset | replay | qualifies |
|---|---|---:|---:|---:|---:|---:|---:|:---:|
| put_the_cream_cheese_in_the_bowl | shift040mm | 20 | 0 | 0.000 | 0.300 | 14.0 | 1.000 | yes |
| put_the_cream_cheese_in_the_bowl | shift060mm | 20 | 0 | 0.000 | 0.650 | 14.0 | 1.000 | yes |
| put_the_bowl_on_the_stove | lateral040mm | 23 | 1 | 0.000 | 0.217 | 10.0 | 1.000 | no |
| put_the_bowl_on_the_stove | lateral050mm | 23 | 1 | 0.000 | 0.087 | 8.5 | 1.000 | no |
| put_the_bowl_on_the_stove | lateral060mm | 23 | 1 | 0.000 | 0.043 | 12.0 | 1.000 | no |
| put_the_bowl_on_the_stove | lateral075mm | 23 | 1 | 0.000 | 0.130 | 8.0 | 1.000 | no |

| task | frozen severity | qualifies | forced diagnostic continuation |
|---|---|:---:|:---:|
| put_the_cream_cheese_in_the_bowl | shift060mm | yes | no |
| put_the_cream_cheese_in_the_bowl | shift040mm | yes | no |
| put_the_bowl_on_the_stove | lateral040mm | no | yes |
| put_the_bowl_on_the_stove | lateral075mm | no | yes |

## Phase D — Full actor-history replay

Status **BLOCKED**；`432` fresh reconstructions / `144` groups。State/history/chunk reconstruction=`0.944444`；contact/outcome agreement=`0.944444`；max state error=`0.0`；no order dependence=`False`。
Failed replay groups=`8`，error records=`24`；affected event IDs=`["put_the_bowl_on_the_stove__seed29__init10"]`；failure signatures=`["ValueError: anchor reconstruction rejected: {'checkpoint_hash_match': True, 'anchor_state_hash_match': False, 'state_history_hash_match': False, 'action_history_hash_match': True, 'original_chunk_hash_match': False, 'max_anchor_state_error_le_1e_9': True}"]`。

## Phase E — Conditional Track A

Track A = **INCONCLUSIVE**。A_original vs B0 的 new-action reduction=`0.379`；两任务 median `(k_last_safe-d)`=`{"put_the_bowl_on_the_stove": 14.0, "put_the_cream_cheese_in_the_bowl": 14.0}`；seed directions=`{"17": true, "29": true, "7": true}`。
Task directions=`{"put_the_bowl_on_the_stove": true, "put_the_cream_cheese_in_the_bowl": false}`；severity directions=`{"lateral040mm": true, "lateral075mm": true, "shift040mm": false, "shift060mm": true}`；checks=`{"both_tasks_median_delay_ge_2": true, "direction_consistent_two_of_three_seeds": true, "new_non_nominal_actions_reduction_ge_15pct": true, "not_one_task_or_seed": false, "safe_success_ci_lower_ge_minus_0_05": true, "same_direction_both_severities": false}`。

该结果只测试两个 late-target-invalidation family；即使有 pilot signal，也不反转 universal kill。

## Phase E — Track B oracle replanability

Formal valid events=`40`，branches=`660`；strongest baseline=`A_original`。Track B = **INCONCLUSIVE**：safe-success difference=`0.175`，cause relative reduction=`0.167`，`k_best!=d`=`0.975`，`k_best!=k_last_safe`=`0.375`，interior best=`0.350`，mean policy-call difference=`-23.25`。
Minimum-data pass=`False`；checks=`{"events_ge_20_each_task": false, "events_ge_5_each_task_seed": false, "events_ge_8_each_task_severity": true}`；invalid Atlas events=`{"put_the_bowl_on_the_stove__seed07__init29": ["branch_error"], "put_the_bowl_on_the_stove__seed29__init27": ["branch_error"], "put_the_bowl_on_the_stove__seed29__init28": ["branch_error"], "put_the_bowl_on_the_stove__seed29__init29": ["branch_error"]}`。
Minimum-data task counts=`{"put_the_bowl_on_the_stove": 19, "put_the_cream_cheese_in_the_bowl": 21}`；task/seed counts=`{"put_the_bowl_on_the_stove/seed17": 10, "put_the_bowl_on_the_stove/seed29": 2, "put_the_bowl_on_the_stove/seed7": 7, "put_the_cream_cheese_in_the_bowl/seed17": 9, "put_the_cream_cheese_in_the_bowl/seed29": 9, "put_the_cream_cheese_in_the_bowl/seed7": 3}`。
Invalid Atlas branch records=`60`；affected event IDs=`["put_the_bowl_on_the_stove__seed07__init29", "put_the_bowl_on_the_stove__seed29__init27", "put_the_bowl_on_the_stove__seed29__init28", "put_the_bowl_on_the_stove__seed29__init29"]`；failure signatures=`["ValueError: anchor reconstruction rejected: {'checkpoint_hash_match': True, 'anchor_state_hash_match': False, 'state_history_hash_match': False, 'action_history_hash_match': True, 'original_chunk_hash_match': False, 'max_anchor_state_error_le_1e_9': False}"]`。

N_oracle 是在完整 realized R(k) 后进行的 oracle protocol 选择，不是 deployable algorithm；相同事件的所有 prefix 使用完全相同 B_post，旧动作同样消耗预算。统计以 event 为独立单位，使用 task-stratified paired bootstrap 10,000 次，不把 prefix rows 当独立样本。

## Phase F — Local repair / operator

Operator branches=`480`，events=`40`；router=`NO_OPERATOR_ROUTING_SIGNAL`，local repair=`NO_SIGNAL`。Unique-safe-win rates=`{"bounded_rollback_replan": 0.0, "cause_specific_local_repair": 0.15, "full_replan": 0.1, "hold_one_step_replan": 0.025}`。所有 operator action 都消耗相同 event budget，没有额外 policy-call allowance。
Full-replan safe-success=`0.492`；local-repair safe-success=`0.400`；local-repair task directions=`{"put_the_bowl_on_the_stove": false, "put_the_cream_cheese_in_the_bowl": true}`；unique-winner tasks=`{"bounded_rollback_replan": [], "cause_specific_local_repair": ["put_the_cream_cheese_in_the_bowl"], "full_replan": ["put_the_bowl_on_the_stove", "put_the_cream_cheese_in_the_bowl"], "hold_one_step_replan": ["put_the_bowl_on_the_stove"]}`。
Phase-F positive labels permitted=`False`；descriptive router criteria met=`False`；descriptive local-repair criteria met=`False`。当上游或最小样本 gate 失败时，即使描述性条件满足也不会发放正标签。

## 提升/降低机理反解

反解直接基于冻结代码路径和同事件真实 branch counterfactual，而非新 idea：收益路径是继续执行仍然 task-directed 的 cached suffix，在 ACT h=1 feedback 前保留进度，因而可能减少新动作和 policy calls；下降路径是相同 suffix 在 replanning 前越过 absorbing misaligned-release / bowl-blocker-contact cause boundary。N_oracle 的额外提升还包含看完全部分支后的选择优势，因此绝不能解释成 learned/deployable replanability。具体事件分类与 action/policy-call 差值见 `artifacts/stage2b/mechanism_audit/counterfactual_cases.jsonl`。
A_original vs B0 真实类别与均值=`{"all_budgets_equal": true, "categories": {"cause_safety_gain": 4, "cause_safety_loss": 3, "equal_outcome_efficiency_gain": 15, "safe_success_gain": 13, "safe_success_loss": 5}, "event_count": 40, "mean_delay_difference": 13.625, "mean_new_action_difference": -57.525, "mean_policy_call_difference": -57.525}`。
N_oracle vs A_original 真实类别与均值=`{"all_budgets_equal": true, "categories": {"no_observed_difference": 33, "safe_success_gain": 7}, "event_count": 40, "mean_delay_difference": -1.7, "mean_new_action_difference": -23.25, "mean_policy_call_difference": -23.25}`。

## Learned / deployable evidence 与 novelty

Stage-2B 没有训练 actor、probe、irreversibility/replanability head 或 router；没有 π0.5、RGB/VLA、world model。Learned/deployable evidence = **none**。Novelty 不高于冻结双审边界 **`N2_ORACLE_PROTOCOL_BOUNDARY_ONLY`**；不得声称 N3/N4、general validation 或 accepted idea。

## 资源、恢复与测试

本轮在开发机 PAI DSW 本地执行，最多 2 个并行 A800 workers；没有外部 PAI/DLC/spot job，因为本地执行可行。没有训练，所以 1,000-step checkpoint/autoresume cadence 不适用。Atlas/operator 以 event 为原子 shard，只跳过带完整 marker 且行数匹配的 event；非空证据不静默覆盖。

测试状态：`PASS`；passed=`30`，failed=`0`，required-contract coverage=`20`。

## 最终字段

- `overall`: **BLOCKED_BY_ACTOR_CONDITIONED_PERTURBATION**
- `track_a_conditional_timing`: **INCONCLUSIVE**
- `track_b_replanability`: **INCONCLUSIVE**
- `operator_router`: **NO_OPERATOR_ROUTING_SIGNAL**
- `stage1b_universal_hypothesis`: **KILLED_IMMUTABLE**
- `accepted`: **false**
