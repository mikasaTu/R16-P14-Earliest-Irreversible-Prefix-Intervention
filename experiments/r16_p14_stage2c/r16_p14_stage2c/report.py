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
    outcome_groups = reverse_matched.get("outcome_groups", {})
    improve = outcome_groups.get("cached_improves", {})
    worsen = outcome_groups.get("cached_worsens", {})
    same = outcome_groups.get("same_safe_outcome", {})
    selection = mechanism.get("crossfit_selection", {})
    activation = mechanism.get("qualification", {}).get("target_shift_activation", {})
    noquery = reverse_matched.get("cached_noquery", {})
    immutable = source.get("immutable_parent", {})
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

    lines = [
        "# R16-P14 Stage-2C：Recoverability-Defined and Compute-Matched Validation",
        "",
        "## 一句话结论",
        "",
        f"最终决策是 **{decision['overall']}**。Track A 为 **{decision['track_a_operator_relative_prefix_reuse']}**，Track B 为 **{decision['track_b_crossfit_replanability']}**。这是完整执行后的诊断结果，不是因为 gate 失败而提前停止；`accepted=false`，新颖性上限仍为 `N2_ORACLE_PROTOCOL_BOUNDARY_ONLY`。",
        "",
        "资格 gate 在查看方法收益之前已失败，因此后续数值不能获得正向标签。它们仍然回答了两个机制问题：等计算量下旧 prefix 与新 prefix 的行为差异，以及恢复 actor 家族下最佳重规划位置是否依赖历史。任何结果都不称为物理不可逆性。",
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
        "- 三个冻结 ACT checkpoint seeds 为 7/17/29；所有分支使用冻结 actor、相同任务 init、相同扰动与预算合同。",
        "",
        "## Phase A：正确性与合同修复",
        "",
        f"历史唯一 replay failure 首次可观察分歧出现在全局 step {root_cause['observation_feature']['first_observable_divergent_global_step']}。四帧历史最大差异为 `{root_cause['four_state_history']['max_abs_difference']:.17g}`，模拟器状态最大差异为 `{root_cause['simulator_state']['max_abs_difference']:.17g}`，ACT chunk 最大差异为 `{root_cause['original_actor_chunk']['max_abs_difference']:.17g}`。分类保持为 `LONG_HORIZON_NUMERICAL_CONTEXT_DRIFT_AMPLIFIED_BY_ACT`。",
        "",
        "另外发现 Stage-2C 诊断 trace 曾调用 `current_observation`，从而刷新 LIBERO observable cache。正式提交把 trace=true/false 的刷新日程改成完全一致；独立 PAI probe 的三个新进程均精确重建。没有放宽容差。旧 Stage-2B stove replay 重新按全部尝试计分为 23/24，而不是丢弃错误后的 1.0。",
        "",
        "其他修复包括：错误进入 replay 分母且 formal cell 要求 `error_count=0`；`S_obs(k)` 吸收单调，任何 0→1 使事件无效；任务进度来自 live BDDL predicate/site/object geometry，不再使用 demo-0 终点。",
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
        "主比较严格匹配：两支在 k 前都调用一次 detection-time ACT、都执行恰好 `k-d` 个 prefix action、都在 k 再调用 ACT、都使用 h=16 tail、相同动作上限/任务 horizon/seed/扰动/generator/evaluation actor。唯一处理差异是执行旧 chunk 的 `A[d:k]` 还是 detection 时新生成 chunk 的前 `k-d` 动作。",
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
            ["baseline", "rows", "safe success", "新动作", "actor calls", "平均 k"],
            _baseline_rows(mechanism),
        ),
        "",
        "## Track A：compute-matched cached prefix",
        "",
        f"Track A evaluation rows={aggregate['track_a_rows']}。cached−fresh safe-success 差为 `{_fmt(a_stats.get('estimate') if a_stats else None)}`，10,000 次 `(task, init_state_id)` cluster bootstrap 95% CI 为 `{_ci(a_stats)}`；平均新动作减少比例为 `{_pct(aggregate['track_a_new_action_reduction_fraction'])}`。两任务 recoverable-window 中位数为 `{aggregate['median_recoverable_windows']}`。",
        "",
        f"所有 prefix 对 compute-matched=`{reverse_matched.get('all_compute_matched')}`、budget-matched=`{reverse_matched.get('all_budget_matched')}`。`CACHED_NOQUERY` 与 matched cached 的安全结果一致率为 `{_pct(noquery.get('same_safe_outcome_rate'))}`，平均少 `{_fmt(noquery.get('mean_actor_call_saving'), 2)}` 次 actor call；这是 secondary deployment evidence，不替代主因果比较。",
        "",
        "## Phase E：operator-relative recoverability",
        "",
        f"atlas rows={aggregate['atlas_rows']}，event boundaries={aggregate['boundary_rows']}，invalid/nonmonotonic events={aggregate['invalid_events']}。`k_last_recoverable` 只对冻结恢复算子族 U 定义；`k_irrev_U` 是该算子族下的 persistent crossing，原始非单调恢复 pattern 完整保留。它不是物理不可逆点。",
        "",
        "## Track B：cross-fitted replanability",
        "",
        f"LOAO cross-fit evaluation rows={aggregate['crossfit_rows']}，calibration-only 冻结最强 baseline 为 `{decision['frozen_strongest_baseline']}`。held-out safe-success 差为 `{_fmt(b_stats.get('estimate') if b_stats else None)}`，95% CI=`{_ci(b_stats)}`；平均新动作减少比例为 `{_pct(aggregate['track_b_new_action_reduction_fraction'])}`。",
        "",
        f"选择分布 `{selection.get('selected_prefix_distribution', {})}`；baseline 分布 `{selection.get('baseline_prefix_distribution', {})}`；与 d 不同 `{_pct(selection.get('differs_from_detection_rate'))}`，与 `k_last_recoverable` 不同 `{_pct(selection.get('differs_from_last_recoverable_rate'))}`，内部 prefix `{_pct(selection.get('interior_prefix_rate'))}`。held-out actor 的 outcome 在选 k 后才读取。",
        "",
        "## 10,000 次分层 cluster bootstrap",
        "",
        "独立 cluster 始终是 `(task, init_state_id)`；actor seed 是 cluster 内重复测量，prefix row 不是独立样本。任务、严重度、baseline、阈值均未用 evaluation outcome 重新选择。",
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
        f"evaluation 中共有 {reverse_matched.get('evaluation_paired_prefix_cells', 0)} 个严格 cached/fresh prefix pair：cached 改善安全结果 {improve.get('count', 0)} 个，恶化 {worsen.get('count', 0)} 个，不变 {same.get('count', 0)} 个。",
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
        "这项反解只把观测差异映射回已冻结代码路径：改善时检查名义进度是否保留、是否没有新增 cause violation；恶化时检查 stale prefix 的 cause violation、cached/fresh action displacement、路径与接触。它不是新 idea，也不是 learned mediator。",
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
        "- learned/deployable evidence 仍为 **NONE**。`CACHED_NOQUERY` 只是冻结 actor 上的离线/模拟 secondary evidence。",
        "- 这不是 VLA、world model、π0.5 或真实机器人实验；没有闭环部署或物理不可逆性证明。",
        "- Stage-1b universal kill 不可逆转；Stage-2B blocked/inconclusive 不被覆盖。",
        "- 最终 `accepted=false`，novelty 不高于 N2。按 Stage-2C 停止点到此结束，不自动进入 learned replanability。",
        "",
        "## 可复现产物",
        "",
        "代码、原始 attempts、actor events、replay traces、全部 matched/recovery/cross-fit rows、10,000-draw 统计、机理审计、测试输出和 SHA256 清单均保存在本仓库的 `experiments/r16_p14_stage2c/` 与 `artifacts/stage2c/`。PAI v1–v8 和独立 replay probe 的每次失败/停止/成功记录均保留，未隐藏基础设施失败。",
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
