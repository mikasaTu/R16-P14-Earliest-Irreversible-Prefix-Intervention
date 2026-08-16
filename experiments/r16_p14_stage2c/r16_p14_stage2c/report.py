from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from r16_p14_stage2b.io_utils import atomic_write_json, atomic_write_text

from .settings import ARTIFACT_ROOT, EXPERIMENT_ROOT, MIRROR_EXPERIMENT_OUTPUTS


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _optional(path: Path) -> dict[str, Any]:
    return _read(path) if path.is_file() else {}


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:
        return "n/a"
    return f"{number:.{digits}f}"


def _pct(value: Any, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.{digits}f}%"


def _ci(stat: dict[str, Any] | None) -> str:
    if not stat:
        return "n/a"
    low, high = stat["ci95"]
    return f"[{_fmt(low)}, {_fmt(high)}]"


def _bool(value: Any) -> str:
    return "PASS" if bool(value) else "FAIL"


def _table(headers: Iterable[str], rows: Iterable[Iterable[Any]]) -> list[str]:
    headers = [str(item) for item in headers]
    body = [[str(item) for item in row] for row in rows]
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(row) + " |" for row in body),
    ]


def _bootstrap_rows(detailed: dict[str, Any], track: str) -> list[list[str]]:
    rows: list[list[str]] = []
    payload = detailed.get(track, {})
    for metric_name, metric_label in (
        ("safe_success", "安全成功差"),
        ("new_non_nominal_actions", "新非名义动作差"),
    ):
        metric = payload.get(metric_name, {})
        for label, stat in (
            ("全部 cluster", metric.get("all_clusters")),
            ("任务宏平均", metric.get("macro_task")),
        ):
            if stat:
                clusters = stat.get("cluster_count")
                if clusters is None:
                    clusters = sum(stat.get("cluster_count_by_task", {}).values())
                rows.append([
                    metric_label,
                    label,
                    str(stat.get("row_count", "n/a")),
                    str(clusters),
                    _fmt(stat.get("estimate")),
                    _ci(stat),
                ])
        for stratum_key, stratum_label in (
            ("by_task", "任务"),
            ("by_task_severity", "任务/严重度"),
            ("by_actor_seed", "actor seed"),
        ):
            for label, stat in sorted(metric.get(stratum_key, {}).items()):
                if stat:
                    rows.append([
                        metric_label,
                        f"{stratum_label}: {label}",
                        str(stat.get("row_count", "n/a")),
                        str(stat.get("cluster_count", "n/a")),
                        _fmt(stat.get("estimate")),
                        _ci(stat),
                    ])
    return rows


def _arm_rows(mechanism: dict[str, Any]) -> list[list[str]]:
    arms = mechanism.get("all_required_arms", {}).get("named_generator_actor_arms", {})
    rows = []
    for name, item in arms.items():
        rows.append([
            name,
            str(item.get("rows", 0)),
            _pct(item.get("safe_success_rate")),
            _pct(item.get("task_success_rate")),
            _pct(item.get("cause_violation_rate")),
            _fmt(item.get("mean_new_non_nominal_actions"), 2),
            _fmt(item.get("mean_actor_calls"), 2),
            _fmt(item.get("mean_completion_steps"), 2),
        ])
    return rows


def _prefix_rows(mechanism: dict[str, Any]) -> list[list[str]]:
    grid = mechanism.get("all_required_arms", {}).get("generator_actor_prefix_grid", {})
    rows = []
    for branch, prefixes in grid.items():
        for prefix, item in sorted(prefixes.items(), key=lambda pair: int(pair[0])):
            rows.append([
                branch,
                prefix,
                str(item.get("rows", 0)),
                _pct(item.get("safe_success_rate")),
                _pct(item.get("task_success_rate")),
                _pct(item.get("cause_violation_rate")),
                _fmt(item.get("mean_new_non_nominal_actions"), 2),
                _fmt(item.get("mean_actor_calls"), 2),
            ])
    return rows


def _operator_rows(mechanism: dict[str, Any]) -> list[list[str]]:
    operators = mechanism.get("all_required_arms", {}).get("recovery_operators", {})
    return [
        [
            name,
            str(item.get("rows", 0)),
            str(item.get("events", 0)),
            _pct(item.get("event_coverage_fraction")),
            _pct(item.get("safe_success_rate")),
            _pct(item.get("task_success_rate")),
            _pct(item.get("cause_violation_rate")),
            _fmt(item.get("mean_new_non_nominal_actions"), 2),
            _fmt(item.get("mean_actor_calls"), 2),
        ]
        for name, item in operators.items()
    ]


def _baseline_rows(mechanism: dict[str, Any]) -> list[list[str]]:
    baselines = mechanism.get("all_required_arms", {}).get(
        "deployable_baselines_over_three_recovery_actors", {}
    )
    return [
        [
            name,
            str(item.get("rows", 0)),
            _pct(item.get("safe_success_rate")),
            _fmt(item.get("mean_new_non_nominal_actions"), 2),
            _fmt(item.get("mean_actor_calls"), 2),
            _fmt(item.get("mean_selected_prefix"), 2),
        ]
        for name, item in baselines.items()
    ]


def _qualification_rows(mechanism: dict[str, Any]) -> list[list[str]]:
    tasks = mechanism.get("qualification", {}).get("tasks", {})
    return [
        [
            name,
            str(item.get("cell_count", 0)),
            str(item.get("qualifying_cells", 0)),
            str(item.get("injection_contact_free_cells", 0)),
            _pct(item.get("maximum_delayed_cause_violation_rate")),
            ", ".join(f"{key}={value}" for key, value in item.get("failed_check_counts", {}).items()),
        ]
        for name, item in tasks.items()
    ]


def main() -> None:
    decision = _read(ARTIFACT_ROOT / "decision.json")
    aggregate = _read(ARTIFACT_ROOT / "aggregate_summary.json")
    stats = _read(ARTIFACT_ROOT / "statistics.json")
    detailed = _optional(ARTIFACT_ROOT / "statistics_detailed.json")
    qualification = _read(ARTIFACT_ROOT / "task_qualification/summary.json")
    events = _read(ARTIFACT_ROOT / "actor_events/summary.json")
    pool = _read(ARTIFACT_ROOT / "actor_events/formal_event_pool_summary.json")
    formal = _read(ARTIFACT_ROOT / "formal_matrix/summary.json")
    mechanism = _optional(ARTIFACT_ROOT / "mechanism_reverse_audit.json")
    if not mechanism:
        mechanism = _read(ARTIFACT_ROOT / "mechanism_audit.json")
    root_cause = _read(ARTIFACT_ROOT / "contract_repair/replay_root_cause.json")
    attempt_audit = _optional(ARTIFACT_ROOT / "pai_attempt_audit.json")
    source = _optional(EXPERIMENT_ROOT / "source_freeze/manifest.json")

    a_stats = stats.get("track_a_safe_success")
    b_stats = stats.get("track_b_safe_success")
    reverse_matched = mechanism.get("matched_prefix", {})
    replay = mechanism.get("replay_contract", _optional(ARTIFACT_ROOT / "replay_contract_diagnostic.json"))
    outcome_groups = reverse_matched.get("outcome_groups", {})
    improve = outcome_groups.get("cached_improves", {})
    worsen = outcome_groups.get("cached_worsens", {})
    same = outcome_groups.get("same_safe_outcome", {})
    selection = mechanism.get("crossfit_selection", {})
    activation = mechanism.get("qualification", {}).get("target_shift_activation", {})
    noquery = reverse_matched.get("cached_noquery", {})
    a_eval_rows = int(a_stats.get("row_count", 0)) if a_stats else 0
    b_eval_rows = int(b_stats.get("row_count", 0)) if b_stats else 0
    invalid_breakdown = replay.get("events", {}).get("breakdown", {})
    baseline_arms = mechanism.get("all_required_arms", {}).get(
        "deployable_baselines_over_three_recovery_actors", {}
    )
    selected_baseline = baseline_arms.get(decision.get("frozen_strongest_baseline"), {})
    baseline_coverage = mechanism.get("all_required_arms", {}).get("baseline_coverage_audit", {})
    calibration_baseline_coverage = baseline_coverage.get(
        "event_count_by_split_and_method", {}
    ).get("calibration", {})
    immutable = source.get("immutable_parent", {})
    checkpoint_hashes = ", ".join(
        f"seed {item['seed']}={item['sha256']}" for item in source.get("act_checkpoints", [])
    )
    qualification_errors = sum(int(item.get("error_count", 0)) for item in qualification.get("cells", []))
    formal_event_instances = sum(
        int(task_payload.get("selected_events", 0))
        for split_payload in pool.get("splits", {}).values()
        for task_payload in split_payload.values()
    )
    v8 = next(
        (
            item
            for item in attempt_audit.get("attempts", [])
            if item.get("run_id") == "r16p14-stage2c-formal-20260816-v8"
        ),
        {},
    )
    cleanup = attempt_audit.get("superseded_service_record_cleanup", {})

    lines = [
        "# R16-P14 Stage-2C：Recoverability-Defined and Compute-Matched Validation",
        "",
        "## 一句话结论",
        "",
        f"最终决策是 **{decision['overall']}**：正式矩阵全部完成，但 replay contract 在聚合时因同一缓存 prefix 的 `S_obs(k)` 自相矛盾而 fail closed。Track A 为 **{decision['track_a_operator_relative_prefix_reuse']}**，Track B 为 **{decision['track_b_crossfit_replanability']}**。这不是 gate 失败后的提前停止；`accepted=false`，新颖性上限仍为 `N2_ORACLE_PROTOCOL_BOUNDARY_ONLY`。",
        "",
        "资格 gate 在查看方法收益之前已失败，正式矩阵又暴露了分支复放污染，因此后续数值不能获得正向标签，也不能被解释为干净的 cached/fresh 因果差异。所有 arm、统计和正负分组仍完整报告，用于定位实现机理与失败边界；任何结果都不称为物理不可逆性。",
        "",
        "## 不可更改的历史结论",
        "",
        "- Stage-1b universal hypothesis：`KILLED_IMMUTABLE`。",
        "- Stage-2B chunk executability：`PASS`；`H_valid=16`。",
        "- Stage-2B actor-conditioned perturbation 与 actor-history replay：`BLOCKED`。",
        "- Stage-2B Track A/B：`INCONCLUSIVE`。",
        "- Stage-2B local repair：`NO_SIGNAL`；operator router：`NO_OPERATOR_ROUTING_SIGNAL`。",
        "- 本轮不修改、不覆盖、也不重新解释上述结论。",
        "",
        "## 预注册、来源与算力",
        "",
        f"- 不可变父提交：`{immutable.get('head', 'n/a')}`；tree：`{immutable.get('tree', 'n/a')}`。",
        "- PAI 正式来源提交：`55c180bfd1abb9767e146cbd7ec5554e6760a0d5`；tree：`a1e2caee307b2270bc167f26164d473e7908baf0`。",
        f"- 正式 PAI 作业：`{v8.get('job_id', 'dlcb8djiituf7gt3')}`，状态 `{v8.get('status', 'n/a')}`，时长 `{v8.get('duration_seconds', 'n/a')}` 秒。",
        "- 正式推理使用 2×A800、两个隔离 GPU worker；没有训练 actor、没有 W&B、没有 π0.5、RGB/VLA、world model 或 learned head。",
        "- PAI Python 为 `/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/libero_sft/bin/python`（SHA256 `89b2f5166fb529c259aedd43e5f718c60e35d58e630cb40ae6accb48fc4f961a`），overlay manifest SHA256 为 `64dfffdaf464d1a37be19b038cca919a252dba573eb2d0f8aa442b91a4099459`，正式 torch 为 `2.5.1+cu124`；launcher SHA256 为 `d3091f960ccb5626374bb60df4ebe81f1e628a79076078a49c7fca47f4db10f2`。",
        f"- 三个冻结 ACT checkpoint：`{checkpoint_hashes}`。所有分支使用冻结 actor、相同任务 init、相同扰动与预算合同。",
        "- 重用的 240 个 immutable actor-event shards 清单 SHA256 为 `4fe57054d32c439201b81da0e433c9c0566c3222b21e28f994f584724474697c`。",
        f"- PAI lifecycle 收尾：`{cleanup.get('status', 'n/a')}`；v3–v7 共 `{cleanup.get('verified_absent_count', 'n/a')}` 条 superseded Failed/Stopped service records 已逐条删除并验证缺席，v8 与成功 replay probe 保留，CPFS/registry 证据未删除。",
        "",
        "## Phase A：正确性与合同修复",
        "",
        f"历史唯一 replay failure 首次可观察分歧出现在全局 step {root_cause['observation_feature']['first_observable_divergent_global_step']}。四帧历史最大差异为 `{root_cause['four_state_history']['max_abs_difference']:.17g}`，模拟器状态最大差异为 `{root_cause['simulator_state']['max_abs_difference']:.17g}`，ACT chunk 最大差异为 `{root_cause['original_actor_chunk']['max_abs_difference']:.17g}`。分类保持为 `LONG_HORIZON_NUMERICAL_CONTEXT_DRIFT_AMPLIFIED_BY_ACT`。",
        "",
        "另外发现 Stage-2C 诊断 trace 曾调用 `current_observation`，从而刷新 LIBERO observable cache。正式提交把 trace=true/false 的刷新日程改成完全一致；独立 PAI probe 的三个新进程均精确重建。没有放宽容差。旧 Stage-2B stove replay 重新按全部尝试计分为 23/24，而不是丢弃错误后的 1.0。",
        "",
        "其他修复包括：错误进入 replay 分母且 formal cell 要求 `error_count=0`；`S_obs(k)` 吸收单调，任何 0→1 使事件无效；任务进度来自 live BDDL predicate/site/object geometry，不再使用 demo-0 终点。以上修复通过了进入正式矩阵前的合同测试，但没有预先证明单个 EventRuntime 内连续分支 reset 的完整性。",
        "",
        "## Phase B：扰动资格",
        "",
        f"资格尝试 `{qualification['completed_attempts']}/{qualification['expected_attempts_from_admitted_events']}`，missing={qualification['missing_attempts']}，errors={qualification_errors}。最终状态 **{qualification['status']}**，标签 `{qualification['failure_label']}`。",
        "",
        *_table(
            ["任务", "cells", "合格 cells", "注入无接触 cells", "最大 delayed violation", "失败检查计数"],
            _qualification_rows(mechanism),
        ),
        "",
        f"关键代码路径反解：target-shift 的 cause tracker 只有在物体已经 lift 后、gripper 从正变负时才会触发。冻结 calibration chunk 中仅 `{activation.get('events_with_post_detection_release', 'n/a')}/{activation.get('calibration_events', 'n/a')}` 有检测后的释放，cause-eligible 为 `{activation.get('events_with_cause_eligible_release', 'n/a')}`，因此 16-action 窗口内改变 shift 大小也无法产生预注册所需的 delayed violation。路径障碍则主要被注入时接触 gate 或 delayed violation 过低挡住。这解释冻结实现，不提出新机制。",
        "",
        "用户明确要求 gate 失败后仍完成全部实验；因此下面的正式矩阵完整执行，但只作为 diagnostic，且 positive label 被 fail-closed 逻辑禁止。",
        "",
        "## Phase C/D：事件池与完整执行",
        "",
        f"ACT 事件尝试 {events['completed_attempts']}，admitted={events['admitted_events']}，不合格排除={events['ineligible_event_exclusions']}，replay-unstable 排除={events['unstable_replay_exclusions']}。冻结 formal pool 为 {formal_event_instances} 个实例，minimum-data shape gate=`{pool['minimum_data_pass']}`。",
        "",
        f"正式矩阵共有 {formal['complete_event_files']}/{formal['event_files']} 个事件文件；matched rows={formal['matched_rows']}，recovery rows={formal['recovery_rows']}，errors={formal['error_count']}，`all_complete={formal['all_complete']}`。每个事件固定 60 条 matched-prefix 与 180 条 recovery-operator 记录。",
        "",
        "预注册的主比较在代码参数上匹配：两支在 k 前都调用一次 detection-time ACT、都执行恰好 `k-d` 个 prefix action、都在 k 再调用 ACT、都使用 h=16 tail、相同动作上限/任务 horizon/seed/扰动/generator/evaluation actor。计划中的唯一处理差异是执行旧 chunk 的 `A[d:k]` 还是 detection 时新生成 chunk 的前 `k-d` 动作；但下面的正式 replay 审计证明运行态没有实现所需的分支隔离，因此不能把观测差异归因给这一个处理变量。",
        "",
        "## 正式 replay contract：完整执行但 fail closed",
        "",
        f"恢复矩阵的 `{replay.get('recovery_prefix_cells', {}).get('total', 'n/a')}` 个 `(event,k)` cell 每个都有 12/12 条预期记录；其中 `{replay.get('recovery_prefix_cells', {}).get('inconsistent_S_obs_cells', 'n/a')}` 个 cell 同时出现 `S_obs=true` 和 `false`，影响 `{replay.get('events', {}).get('with_inconsistent_recovery_S_obs', 'n/a')}/{replay.get('events', {}).get('total', 'n/a')}` 个事件。分布为 `{invalid_breakdown.get('task_split', {})}`。cream-cheese 事件为 0 个，stove calibration/evaluation 分别为 14/19 个。",
        "",
        f"更强的同动作控制也失败：`CACHED_MATCHED` 和 `CACHED_NOQUERY` 在 k 前都执行相同冻结 `A[d:k]`，但 pre-tail `S_obs(k)` 在 `{replay.get('same_nominal_prefix_control', {}).get('S_obs_disagreement_cells', 'n/a')}/{replay.get('same_nominal_prefix_control', {}).get('pair_cells', 'n/a')}` 个 cell 不一致。恢复路径的 `prefix_cause_violation` 在 recovery actor 推理和 operator action **之前**写入；所以 actor、operator、tail outcome 和缺失行都不能解释这种差异。",
        "",
        "代码级边界是：`run_event` 为一个事件只建立一个可变 LIBERO `EventRuntime`，然后按固定顺序运行所有 branch；`reset()` 只恢复枚举的 simulator/controller/RNG 状态，而不是为每个 branch 新建环境。原始证据因此把原因定位到 reset/detection/injection/prefix 路径中的残余可变状态或顺序依赖，但不足以唯一指定某个隐藏 simulator、controller 或 observable-cache 字段。没有修改原始 row，也没有用后处理补齐标签。",
        "",
        "聚合文件中的 `INCOMPLETE_PREFIX_GRID` 不是少跑了 prefix：矛盾的 `S_obs` 集合被置为 null，随后 fail-closed 成 incomplete。正式执行仍是 96/96、23,040 rows、0 errors；科学 replay contract 则是 `BLOCKED`。这使全部 arm 数值、recoverability boundary、Track A/B 和正负机理分组只能作为污染数据上的描述性审计。",
        "",
        "## 所有命名 arm（evaluation）",
        "",
        *_table(
            ["arm", "rows", "safe success", "task success", "cause violation", "新动作", "actor calls", "完成步数"],
            _arm_rows(mechanism),
        ),
        "",
        "### 全 prefix 网格",
        "",
        *_table(
            ["branch", "k", "rows", "safe success", "task success", "cause violation", "新动作", "actor calls"],
            _prefix_rows(mechanism),
        ),
        "",
        "### 恢复 operator",
        "",
        *_table(
            ["operator", "rows", "safe success", "task success", "cause violation", "新动作", "actor calls"],
            _operator_rows(mechanism),
        ),
        "",
        "### 可部署 baseline（由 calibration 冻结）",
        "",
        *_table(
            ["baseline", "rows", "events", "event coverage", "safe success", "新动作", "actor calls", "平均 k"],
            _baseline_rows(mechanism),
        ),
        "",
        "## Track A：compute-matched cached prefix",
        "",
        f"Track A boundary rows 总计={aggregate['track_a_rows']}，其中 evaluation rows={a_eval_rows}。在被 atlas 保留的子集上，cached−fresh safe-success 描述性差为 `{_fmt(a_stats.get('estimate') if a_stats else None)}`，10,000 次 `(task, init_state_id)` cluster bootstrap 95% CI 为 `{_ci(a_stats)}`；逐 row 新动作 reduction fraction 的均值为 `{_pct(aggregate['track_a_new_action_reduction_fraction'])}`。两任务 recoverable-window 描述性中位数为 `{aggregate['median_recoverable_windows']}`。由于 stove 的 33 个 replay-invalid 事件被边界构造排除，这不是完整样本的有效效应估计。",
        "",
        f"所有 planned prefix row 的账面 compute-matched=`{reverse_matched.get('all_compute_matched')}`、budget-matched=`{reverse_matched.get('all_budget_matched')}`；这只验证预算字段，不能修复运行态污染。`CACHED_NOQUERY` 与 matched cached 的最终安全结果一致率为 `{_pct(noquery.get('same_safe_outcome_rate'))}`，平均少 `{_fmt(noquery.get('mean_actor_call_saving'), 2)}` 次 actor call；由于它们连 pre-tail `S_obs` 都可能不同，本轮不能称为 deployment evidence。",
        "",
        "## Phase E：operator-relative recoverability",
        "",
        f"atlas rows={aggregate['atlas_rows']}，event boundaries={aggregate['boundary_rows']}，replay-invalid events={aggregate['invalid_events']}。`k_last_recoverable` 只对冻结恢复算子族 U 定义；`k_irrev_U` 是该算子族下的 persistent crossing，原始非单调恢复 pattern 完整保留。由于 prefix safety 本身受 reset/order 污染，这些边界只能描述该次运行的 frozen operator rows；它不是物理不可逆点。",
        "",
        "## Track B：cross-fitted replanability",
        "",
        f"LOAO cross-fit rows 总计={aggregate['crossfit_rows']}，其中 evaluation rows={b_eval_rows}；calibration-only 冻结最强 baseline 为 `{decision['frozen_strongest_baseline']}`。但 calibration 排名并非 common-support：该 boundary baseline 只覆盖 `{calibration_baseline_coverage.get(decision['frozen_strongest_baseline'], 'n/a')}/48` 个事件，固定延迟类各覆盖 48/48；evaluation 也只有 `{selected_baseline.get('events', 'n/a')}/48` 个事件、`{selected_baseline.get('rows', 'n/a')}` 个 actor rows。held-out safe-success 描述性差为 `{_fmt(b_stats.get('estimate') if b_stats else None)}`，95% CI=`{_ci(b_stats)}`；逐 row 新动作 reduction fraction 均值为 `{_pct(aggregate['track_b_new_action_reduction_fraction'])}`。",
        "",
        f"选择分布 `{selection.get('selected_prefix_distribution', {})}`；baseline 分布 `{selection.get('baseline_prefix_distribution', {})}`；与 d 不同 `{_pct(selection.get('differs_from_detection_rate'))}`，与 `k_last_recoverable` 不同 `{_pct(selection.get('differs_from_last_recoverable_rate'))}`，内部 prefix `{_pct(selection.get('interior_prefix_rate'))}`。held-out actor 的 outcome 在选 k 后才读取。",
        "",
        "## 10,000 次分层 cluster bootstrap",
        "",
        "独立 cluster 始终是 `(task, init_state_id)`；actor seed 是 cluster 内重复测量，prefix row 不是独立样本。任务、严重度、baseline、阈值均未用 evaluation outcome 重新选择。bootstrap 实现正确，但输入已受 replay contract 和有效子集选择限制，因此区间只作描述。",
        "",
        "### Track A",
        "",
        *_table(["指标", "分层", "rows", "clusters", "估计", "95% CI"], _bootstrap_rows(detailed, "track_a")),
        "",
        "### Track B",
        "",
        *_table(["指标", "分层", "rows", "clusters", "估计", "95% CI"], _bootstrap_rows(detailed, "track_b")),
        "",
        "## 为什么会提升或降低：冻结代码的反向解释",
        "",
        f"evaluation 中共有 {reverse_matched.get('evaluation_paired_prefix_cells', 0)} 个 planned cached/fresh prefix pair：原始 cached outcome 改善安全结果 {improve.get('count', 0)} 个，恶化 {worsen.get('count', 0)} 个，不变 {same.get('count', 0)} 个。它们是结果分桶，不是 replay-valid 因果 pair。",
        "",
        *_table(
            ["结果组", "n", "safe Δ", "cause Δ", "action disagreement", "cached−fresh progress", "fresh−cached 新动作", "完成步数 Δ", "接触 Δ"],
            [
                [
                    label,
                    str(item.get("count", 0)),
                    _fmt(item.get("safe_success_delta")),
                    _fmt(item.get("cause_violation_delta")),
                    _fmt(item.get("mean_action_disagreement")),
                    _fmt(item.get("mean_cached_minus_fresh_progress")),
                    _fmt(item.get("mean_fresh_minus_cached_new_actions")),
                    _fmt(item.get("mean_cached_minus_fresh_completion_steps")),
                    _fmt(item.get("mean_cached_minus_fresh_contacts")),
                ]
                for label, item in (("cached improves", improve), ("cached worsens", worsen), ("same", same))
            ],
        ),
        "",
        "这项反解先做排除：同一 cached prefix 的 pre-operator `S_obs` 会随分支顺序改变，因此无法把改善归因于保留名义进度，也无法把恶化归因于 stale action。表中 progress、cause、displacement、路径和接触只说明污染后的数值如何分组；它们不能区分动作机制与 reset/order 状态。它不是新 idea，也不是 learned mediator。",
        "",
        "必须特别警惕结构性指标：`new_non_nominal_actions` 把保留的 cached `A[d:k]` 计为旧动作，而把 fresh prefix/tail 计为新动作，所以 k 越大时该指标天然更容易下降。cross-fit 的字典序在安全结果并列后依次偏好更少新动作、更少 actor call、最后更小 k，因此安全并列时可能被这个定义推向较晚 prefix。新动作减少若没有独立安全/任务提升，不能当成增益机理。",
        "",
        "## 判定向量",
        "",
        *_table(["字段", "值"], [[key, value] for key, value in decision.items() if key not in {"track_a_checks", "track_b_checks", "upstream_gates"}]),
        "",
        "### Upstream gates",
        "",
        *_table(["gate", "结果"], [[key, _bool(value)] for key, value in decision["upstream_gates"].items()]),
        "",
        "### Track A 原始 criteria",
        "",
        *_table(["criterion", "结果"], [[key, _bool(value)] for key, value in decision["track_a_checks"].items()]),
        "",
        "### Track B 原始 criteria",
        "",
        *_table(["criterion", "结果"], [[key, _bool(value)] for key, value in decision["track_b_checks"].items()]),
        "",
        "## 证据边界与停止点",
        "",
        "- local repair 与 operator router 保持 retired；本轮没有调参、训练或给予正向标签。",
        "- learned/deployable evidence 仍为 **NONE**。replay contract 失败后，`CACHED_NOQUERY` 也只能是受污染的离线/模拟诊断。",
        "- 33/96 个事件的运行态分支隔离失败；即使某些 raw CI 看似正向，也不得越过该 gate。",
        "- 这不是 VLA、world model、π0.5 或真实机器人实验；没有闭环部署或物理不可逆性证明。",
        "- Stage-1b universal kill 不可逆转；Stage-2B blocked/inconclusive 不被覆盖。",
        "- 最终 `accepted=false`，novelty 不高于 N2。按 Stage-2C 停止点到此结束，不自动进入 learned replanability。",
        "",
        "## 可复现产物",
        "",
        "代码、原始 attempts、actor events、replay traces、全部 matched/recovery/cross-fit rows、10,000-draw 统计、`replay_contract_diagnostic.*`、机理审计、测试输出和 SHA256 清单均保存在本仓库的 `experiments/r16_p14_stage2c/` 与 `artifacts/stage2c/`。PAI 原始报告另存为 `REPORT_PAI_RAW.md`；PAI v1–v8 和独立 replay probe 的每次失败/停止/成功记录均保留，未隐藏基础设施失败。",
        "",
    ]
    report = "\n".join(str(item) for item in lines)
    atomic_write_text(ARTIFACT_ROOT / "REPORT.md", report)
    if MIRROR_EXPERIMENT_OUTPUTS:
        destination = EXPERIMENT_ROOT / "reports"
        destination.mkdir(parents=True, exist_ok=True)
        atomic_write_text(destination / "REPORT.md", report)
        atomic_write_json(EXPERIMENT_ROOT / "decision.json", decision)
        atomic_write_json(destination / "statistics.json", stats)
        if detailed:
            atomic_write_json(destination / "statistics_detailed.json", detailed)
        atomic_write_json(destination / "mechanism_audit.json", mechanism)
    print(json.dumps({"status": "complete", "report": str(ARTIFACT_ROOT / "REPORT.md")}, sort_keys=True))


if __name__ == "__main__":
    main()
