from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from .io_utils import atomic_write_json, atomic_write_text
from .settings import ARTIFACT_ROOT, EXPERIMENT_ROOT, PRIMARY_TASKS, PROJECT_ROOT


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True).strip()


def overall_decision(
    phase_a: dict[str, Any],
    events: dict[str, Any],
    phase_c: dict[str, Any],
    phase_d: dict[str, Any],
    atlas: dict[str, Any],
) -> str:
    if phase_a["status"] != "PASS":
        return "BLOCKED_BY_CHUNK_EXECUTABILITY"
    if events.get("event_split_blocked"):
        return "BLOCKED_BY_EVENT_SPLIT"
    if phase_c["status"] != "PASS":
        return "BLOCKED_BY_ACTOR_CONDITIONED_PERTURBATION"
    if phase_d["status"] != "PASS":
        return "BLOCKED_BY_ACTOR_HISTORY_REPLAY"
    track_a = atlas["track_a"]["status"] == "PILOT_SIGNAL"
    track_b = atlas["track_b"]["status"] == "PILOT_SIGNAL"
    if track_a and track_b:
        return "PROCEED_TO_FULL_ATLAS_WITH_BOTH_TRACKS"
    if track_b:
        return "PROCEED_TO_FULL_ATLAS_REPLANABILITY_ONLY"
    if track_a:
        return "PROCEED_TO_FULL_ATLAS_CONDITIONAL_TIMING_ONLY"
    return "STOP_PREFIX_TIMING_FAMILY"


def main() -> None:
    phase_a = read(ARTIFACT_ROOT / "chunk_executability/summary.json")
    events = read(ARTIFACT_ROOT / "actor_events/summary.json")
    phase_c = read(ARTIFACT_ROOT / "perturbation_qualification/summary.json")
    frozen = read(ARTIFACT_ROOT / "perturbation_qualification/frozen_parameters.json")
    phase_d = read(ARTIFACT_ROOT / "replay/summary.json")
    replay_rows = [
        json.loads(line)
        for line in (ARTIFACT_ROOT / "replay/raw_reconstructions.jsonl").read_text().splitlines()
        if line.strip()
    ]
    replay_error_rows = [row for row in replay_rows if row.get("error") is not None]
    replay_error_events = sorted({row["event_id"] for row in replay_error_rows})
    replay_error_signatures = sorted({row["error"] for row in replay_error_rows})
    atlas = read(ARTIFACT_ROOT / "atlas_pilot/summary.json")
    atlas_rows = [
        json.loads(line)
        for line in (ARTIFACT_ROOT / "atlas_pilot/raw_branches.jsonl").read_text().splitlines()
        if line.strip()
    ]
    atlas_error_rows = [row for row in atlas_rows if row.get("error") is not None]
    atlas_error_events = sorted({row["event_id"] for row in atlas_error_rows})
    atlas_error_signatures = sorted({row["error"] for row in atlas_error_rows})
    readouts = read(ARTIFACT_ROOT / "atlas_pilot/readouts.json")
    operator = read(ARTIFACT_ROOT / "operator_audit/summary.json")
    mechanism = read(ARTIFACT_ROOT / "mechanism_audit/summary.json")
    source = read(ARTIFACT_ROOT / "source_freeze/source_manifest.json")
    tests_path = ARTIFACT_ROOT / "tests/summary.json"
    tests = read(tests_path) if tests_path.is_file() else {"status": "NOT_RECORDED"}
    overall = overall_decision(phase_a, events, phase_c, phase_d, atlas)
    failed_replay_groups = [group for group in phase_d["groups"] if not group["passed"]]
    decision = {
        "schema_version": 1,
        "stage": "R16-P14 Stage-2B",
        "stage1b_universal_hypothesis": "KILLED_IMMUTABLE",
        "stage2a_checkpoint": "PASS_READY_FOR_BOUNDED_ATLAS",
        "chunk_executability": phase_a["status"],
        "H_valid": phase_a["H_valid"],
        "actor_conditioned_perturbation": phase_c["status"],
        "actor_history_replay": phase_d["status"],
        "track_a_conditional_timing": atlas["track_a"]["status"],
        "track_a_local_repair": operator["track_a_local_repair"],
        "track_b_replanability": atlas["track_b"]["status"],
        "operator_router": operator["operator_router"],
        "overall": overall,
        "full_stage2b_experiments_completed": True,
        "upstream_gate_early_stop_overridden_by_user": True,
        "post_registration_measurement_repair": "PHASE_A_TARGET_AWARE_DROP_V2_DISCLOSED",
        "downstream_results_descriptive_when_upstream_blocked": not atlas["all_upstream_gates_pass"],
        "novelty_boundary": "N2_ORACLE_PROTOCOL_BOUNDARY_ONLY",
        "learned_or_deployable_evidence": "NONE",
        "accepted": False,
    }
    atomic_write_json(EXPERIMENT_ROOT / "decision.json", decision)
    atomic_write_json(ARTIFACT_ROOT / "decision.json", decision)
    evidence_commit = git("rev-parse", "HEAD")
    evidence_tree = git("rev-parse", "HEAD^{tree}")
    lines = [
        "# R16-P14 Stage-2B Action-Chunk Faithfulness and Actor-Conditioned Replanability Pilot",
        "",
        "## 结论摘要",
        "",
        f"最终枚举为 **`{overall}`**。本轮按用户最新指令完成 Phase A–F；即使 A–D gate 失败也继续执行后续矩阵，但 gate 失败后的 Atlas/Operator 结果只作描述，不获得正标签。`accepted=false`。",
        "",
        f"- Action-chunk executability: **{phase_a['status']}**, `H_valid={phase_a['H_valid']}`（要求 >=8）。",
        f"- Actor-conditioned perturbation: **{phase_c['status']}**。",
        f"- Full actor-history replay: **{phase_d['status']}**。",
        f"- Track A conditional timing: **{atlas['track_a']['status']}**。",
        f"- Track B oracle replanability: **{atlas['track_b']['status']}**。",
        f"- Local repair: **{operator['track_a_local_repair']}**；operator router: **{operator['operator_router']}**。",
        "",
        "## 不可变负结论与证据冻结",
        "",
        "Stage-1b universal hypothesis 仍为 **`KILLED_IMMUTABLE`**；本轮不能删除、弱化或反转它。Stage-2A checkpoint 仍为 **`PASS_READY_FOR_BOUNDED_ATLAS`**，当时 Track A/B 均为 INCONCLUSIVE，且未运行 full atlas。",
        "",
        f"- Stage-1 actual resolvable commit/tree: `{source['stage1']['commit']}` / `{source['stage1']['tree']}`。",
        f"- Stage-1b commit/tree: `{source['stage1b']['commit']}` / `{source['stage1b']['tree']}`。",
        f"- Stage-2A raw evidence commit/tree: `{source['stage2a_evidence']['commit']}` / `{source['stage2a_evidence']['tree']}`。",
        f"- Stage-2A report commit/tree: `{source['stage2a_report']['commit']}` / `{source['stage2a_report']['tree']}`。",
        f"- Step4 pre-report evidence HEAD/tree: `{evidence_commit}` / `{evidence_tree}`。",
        "- Stage-2A manifest 中旧的 Stage-1 引用 `a1b611...` 在当前仓库不可解析；本轮未掩盖该事实，冻结文件将它标为 historical-unresolvable，并采用当前可验证发布对象 `ee56aa...`。",
        "",
        "## Phase A — Action-chunk executability",
        "",
        f"共运行 `{phase_a['episode_count']}` 个 clean episodes；h=1 aggregate success=`{phase_a['h1_aggregate_success']:.3f}`，late reach=`{phase_a['h1_aggregate_late_phase_reach']:.3f}`。",
        "",
        "| h | success | late reach | prefix faithful | late anchors | errors | pass |",
        "|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for horizon in sorted(phase_a["horizons"], key=int):
        row = phase_a["horizons"][horizon]
        lines.append(
            f"| {horizon} | {row['clean_success_rate']:.3f} | {row['late_phase_reach_rate']:.3f} | "
            f"{row['prefix_faithful_rate']:.3f} | {row['late_anchor_count']} | "
            f"{row['simulator_or_action_contract_error_count']} | {'yes' if row['passes_H_valid_contract'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "Phase A 使用 target-aware object-drop monitor v2。这是一项透明的 post-registration measurement repair：首轮完整 aggregate 将 stove 的正确 placement descent 误标为 drop；旧证据已按 SHA256 隔离，修复发生在 Phase B 和任何扰动/replan outcome 之前。详见 `reports/phase_a_monitor_bug_audit.md`。",
            "",
            "若该 gate 被阻断，它只说明冻结 ACT substrate 不适于 prefix 实验，不解释为 R16-P14 mechanism failure。",
            "",
            "## Phase B — Actor-generated events",
            "",
            f"从 `{events['attempt_count']}` 个 ACT h=1 closed-loop attempts 构建 `{events['event_count']}` 个 eligible events；formal nominal 全部来自 actor chunk，demonstration nominal count=`{events['demonstration_nominal_chunk_count']}`。Reserved IDs 30–49 未 rollout/未查看 outcome。",
            "",
            "| task | eligible total | qualification | calibration | evaluation |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for task in PRIMARY_TASKS:
        row = events["tasks"][task]
        split = row["by_split"]
        lines.append(
            f"| {task} | {row['total']} | {split.get('actor_qualification', 0)} | "
            f"{split.get('perturbation_calibration', 0)} | {split.get('atlas_pilot_evaluation', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Phase C — Actor-conditioned perturbation qualification",
            "",
            "选择只读取 phase validity、immediate/delayed cause、first offset 与 replay；没有读取 immediate-replan、last-safe、k_best 或 method gain。",
            "",
            "| task | tested severity | valid | errors | immediate | delayed | median offset | replay | qualifies |",
            "|---|---|---:|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for task in PRIMARY_TASKS:
        for row in phase_c["task_summaries"][task]:
            lines.append(
                f"| {task} | {row['severity_id']} | {row['valid_event_count']} | {row['error_count']} | "
                f"{row['immediate_cause_violation_rate']:.3f} | {row['delayed_nominal_cause_violation_rate']:.3f} | "
                f"{row['median_first_violation_offset_after_d']} | {row['replay_rate']:.3f} | "
                f"{'yes' if row['qualifies'] else 'no'} |"
            )
    lines.extend(
        [
            "",
            "| task | frozen severity | qualifies | forced diagnostic continuation |",
            "|---|---|:---:|:---:|",
        ]
    )
    for task in PRIMARY_TASKS:
        task_row = frozen["tasks"][task]
        for severity in task_row["severities"]:
            lines.append(
                f"| {task} | {severity['severity_id']} | {'yes' if severity['qualifies'] else 'no'} | "
                f"{'yes' if task_row['forced_for_continuation'] else 'no'} |"
            )
    lines.extend(
        [
            "",
            "## Phase D — Full actor-history replay",
            "",
            f"Status **{phase_d['status']}**；`{phase_d['reconstruction_record_count']}` fresh reconstructions / `{phase_d['comparison_group_count']}` groups。State/history/chunk reconstruction=`{phase_d['state_history_chunk_reconstruction_rate']:.6f}`；contact/outcome agreement=`{phase_d['contact_outcome_agreement']:.6f}`；max state error=`{phase_d['max_numerical_state_error']}`；no order dependence=`{phase_d['no_branch_order_dependence']}`。",
            f"Failed replay groups=`{len(failed_replay_groups)}`，error records=`{len(replay_error_rows)}`；affected event IDs=`{json.dumps(replay_error_events, sort_keys=True)}`；failure signatures=`{json.dumps(replay_error_signatures, sort_keys=True)}`。",
            "",
            "## Phase E — Conditional Track A",
            "",
            f"Track A = **{atlas['track_a']['status']}**。A_original vs B0 的 new-action reduction=`{atlas['track_a']['new_non_nominal_actions_reduction']:.3f}`；两任务 median `(k_last_safe-d)`=`{json.dumps(atlas['track_a']['median_k_last_safe_minus_d'], sort_keys=True)}`；seed directions=`{json.dumps(atlas['track_a']['seed_direction'], sort_keys=True)}`。",
            f"Task directions=`{json.dumps(atlas['track_a']['task_direction'], sort_keys=True)}`；severity directions=`{json.dumps(atlas['track_a']['severity_direction'], sort_keys=True)}`；checks=`{json.dumps(atlas['track_a']['checks'], sort_keys=True)}`。",
            "",
            "该结果只测试两个 late-target-invalidation family；即使有 pilot signal，也不反转 universal kill。",
            "",
            "## Phase E — Track B oracle replanability",
            "",
            f"Formal valid events=`{atlas['valid_event_count']}`，branches=`{atlas['formal_branch_count']}`；strongest baseline=`{atlas['strongest_baseline']}`。Track B = **{atlas['track_b']['status']}**：safe-success difference=`{atlas['track_b']['safe_success_difference']:.3f}`，cause relative reduction=`{atlas['track_b']['cause_violation_relative_reduction']:.3f}`，`k_best!=d`=`{atlas['track_b']['k_best_not_d_rate']:.3f}`，`k_best!=k_last_safe`=`{atlas['track_b']['k_best_not_k_last_safe_rate']:.3f}`，interior best=`{atlas['track_b']['interior_k_best_rate']:.3f}`，mean policy-call difference=`{atlas['track_b']['mean_policy_call_difference']}`。",
            f"Minimum-data pass=`{atlas['minimum_valid_data']['passed']}`；checks=`{json.dumps(atlas['minimum_valid_data']['checks'], sort_keys=True)}`；invalid Atlas events=`{json.dumps(atlas['invalid_events'], sort_keys=True)}`。",
            f"Minimum-data task counts=`{json.dumps(atlas['minimum_valid_data']['task_counts'], sort_keys=True)}`；task/seed counts=`{json.dumps(atlas['minimum_valid_data']['task_seed_counts'], sort_keys=True)}`。",
            f"Invalid Atlas branch records=`{len(atlas_error_rows)}`；affected event IDs=`{json.dumps(atlas_error_events, sort_keys=True)}`；failure signatures=`{json.dumps(atlas_error_signatures, sort_keys=True)}`。",
            "",
            "N_oracle 是在完整 realized R(k) 后进行的 oracle protocol 选择，不是 deployable algorithm；相同事件的所有 prefix 使用完全相同 B_post，旧动作同样消耗预算。统计以 event 为独立单位，使用 task-stratified paired bootstrap 10,000 次，不把 prefix rows 当独立样本。",
            "",
            "## Phase F — Local repair / operator",
            "",
            f"Operator branches=`{operator['record_count']}`，events=`{operator['event_count']}`；router=`{operator['operator_router']}`，local repair=`{operator['track_a_local_repair']}`。Unique-safe-win rates=`{json.dumps(operator['unique_safe_win_event_rates'], sort_keys=True)}`。所有 operator action 都消耗相同 event budget，没有额外 policy-call allowance。",
            f"Full-replan safe-success=`{operator['full_replan_safe_success_rate']:.3f}`；local-repair safe-success=`{operator['local_repair_safe_success_rate']:.3f}`；local-repair task directions=`{json.dumps(operator['local_repair_task_direction'], sort_keys=True)}`；unique-winner tasks=`{json.dumps(operator['unique_winner_tasks'], sort_keys=True)}`。",
            f"Phase-F positive labels permitted=`{operator['positive_labels_permitted']}`；descriptive router criteria met=`{operator['descriptive_operator_router_criteria_met']}`；descriptive local-repair criteria met=`{operator['descriptive_local_repair_criteria_met']}`。当上游或最小样本 gate 失败时，即使描述性条件满足也不会发放正标签。",
            "",
            "## 提升/降低机理反解",
            "",
            "反解直接基于冻结代码路径和同事件真实 branch counterfactual，而非新 idea：收益路径是继续执行仍然 task-directed 的 cached suffix，在 ACT h=1 feedback 前保留进度，因而可能减少新动作和 policy calls；下降路径是相同 suffix 在 replanning 前越过 absorbing misaligned-release / bowl-blocker-contact cause boundary。N_oracle 的额外提升还包含看完全部分支后的选择优势，因此绝不能解释成 learned/deployable replanability。具体事件分类与 action/policy-call 差值见 `artifacts/stage2b/mechanism_audit/counterfactual_cases.jsonl`。",
            f"A_original vs B0 真实类别与均值=`{json.dumps(mechanism['atlas_counterfactual_summary']['A_original_vs_B0'], sort_keys=True)}`。",
            f"N_oracle vs A_original 真实类别与均值=`{json.dumps(mechanism['atlas_counterfactual_summary']['N_oracle_vs_A_original'], sort_keys=True)}`。",
            "",
            "## Learned / deployable evidence 与 novelty",
            "",
            "Stage-2B 没有训练 actor、probe、irreversibility/replanability head 或 router；没有 π0.5、RGB/VLA、world model。Learned/deployable evidence = **none**。Novelty 不高于冻结双审边界 **`N2_ORACLE_PROTOCOL_BOUNDARY_ONLY`**；不得声称 N3/N4、general validation 或 accepted idea。",
            "",
            "## 资源、恢复与测试",
            "",
            "本轮在开发机 PAI DSW 本地执行，最多 2 个并行 A800 workers；没有外部 PAI/DLC/spot job，因为本地执行可行。没有训练，所以 1,000-step checkpoint/autoresume cadence 不适用。Atlas/operator 以 event 为原子 shard，只跳过带完整 marker 且行数匹配的 event；非空证据不静默覆盖。",
            "",
            f"测试状态：`{tests.get('status')}`；passed=`{tests.get('passed')}`，failed=`{tests.get('failed')}`，required-contract coverage=`{tests.get('required_contract_count')}`。",
            "",
            "## 最终字段",
            "",
            f"- `overall`: **{overall}**",
            f"- `track_a_conditional_timing`: **{atlas['track_a']['status']}**",
            f"- `track_b_replanability`: **{atlas['track_b']['status']}**",
            f"- `operator_router`: **{operator['operator_router']}**",
            "- `stage1b_universal_hypothesis`: **KILLED_IMMUTABLE**",
            "- `accepted`: **false**",
            "",
        ]
    )
    report = "\n".join(lines)
    atomic_write_text(EXPERIMENT_ROOT / "reports/REPORT.md", report)
    atomic_write_text(ARTIFACT_ROOT / "REPORT.md", report)
    summary = {
        "decision": decision,
        "evidence_commit": evidence_commit,
        "evidence_tree": evidence_tree,
        "source_freeze": source,
        "tests": tests,
        "reports": {
            "full": "experiments/r16_p14_stage2b/reports/REPORT.md",
            "mechanism": "experiments/r16_p14_stage2b/reports/mechanism_audit.md",
        },
    }
    atomic_write_json(EXPERIMENT_ROOT / "reports/experiment_summary.json", summary)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
