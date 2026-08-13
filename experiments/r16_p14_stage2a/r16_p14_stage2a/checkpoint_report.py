from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from .io_utils import atomic_write_json, atomic_write_text
from .settings import TASK_NAMES


TASK_LABELS = {
    "put_the_cream_cheese_in_the_bowl": "cream-cheese → bowl",
    "put_the_bowl_on_the_plate": "bowl → plate",
    "open_the_middle_drawer_of_the_cabinet": "open middle drawer",
    "open_the_top_drawer_and_put_the_bowl_inside": "open top drawer + bowl inside",
    "push_the_plate_to_the_front_of_the_stove": "push plate → stove front",
    "put_the_bowl_on_the_stove": "bowl → stove",
}
NEXT_COMMAND = "./experiments/r16_p14_stage2a/commands.sh atlas-preflight"


def load(path: Path) -> dict[str, Any] | list[dict[str, Any]]:
    return json.loads(path.read_text())


def git_value(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def render(context: dict[str, Any]) -> str:
    actor = context["actor"]
    perturbations = context["perturbations"]
    replay = context["replay"]
    preflight = context["preflight"]
    roster = context["roster"]
    mechanism = context["mechanism"]
    training = context["training"]
    source = context["source"]
    lines = [
        "# R16-P14 Stage-2A First Checkpoint 实验报告",
        "",
        "## Checkpoint 结论",
        "",
        "**`FIRST_CHECKPOINT_PASS_READY_FOR_BOUNDED_ATLAS`**",
        "",
        "本检查点已完成计划要求的 Phase 0 线索复现、六任务结构筛选、固定扰动网格校准、ACT 训练 smoke、三 seed clean/phase gate，以及正负锚点 fresh-environment replay gate。Actor gate 为 PASS，replay gate 为 PASS；但扰动资格仅有 2/6 任务通过，因此后续 Atlas 必须限制在这两个任务，不能用 evaluation/held-out 结果返调另外四个负任务。完整 Atlas 尚未启动，本报告不对 Track A、Track B 或最终 overall 作结论。",
        "",
        "## 不可变边界与精确快照",
        "",
        f"- 本 checkpoint 代码与原始证据 commit：`{context['head']}`。",
        f"- 对应 Git tree：`{context['tree']}`。",
        f"- 分支：`{context['branch']}`；remote：`{context['remote']}`。",
        f"- 冻结 Stage-1 commit/tree：`{source['stage1_frozen']['commit']}` / `{source['stage1_frozen']['tree']}`。",
        f"- 冻结 Stage-1b commit/tree：`{source['stage1b_frozen']['commit']}` / `{source['stage1b_frozen']['tree']}`。",
        f"- Stage-1b 决策保持 **`{source['stage1b_frozen']['decision']}` / `KILLED_IMMUTABLE`**，本阶段不能撤销。",
        f"- Stage-1b REPORT SHA256：`{source['immutable_file_sha256']['stage1b_report']}`；decision SHA256：`{source['immutable_file_sha256']['stage1b_decision']}`。",
        "- `accepted=false`；未训练 learned irreversibility/replanability head；未使用 π0.5、RGB、DINO-WM 或 world model。",
        "",
        "说明：报告自身在上述证据 commit 之后生成，因此交付文档 commit 会不同；上述 commit/tree 是包含全部实验代码、权重和原始结果的可复现实验快照。",
        "",
        "## Phase 0 — 当前线索机械复现",
        "",
        "从冻结的 `selected_config_paired_metrics.csv` 得到 M0=5/6、M1=5/6、M2=2/6；M0 与 M1 的失败样本不同，二者逐样本 safe union 为 6/6；当 M0、M1 同时失败时，M2 独有 safe success 为 0。",
        "",
        "**这是 calibration-only、n=6、hypothesis-generating 证据，不是算法性能。**",
        "",
        f"输入 SHA256：`{context['clue']['source_sha256']}`。",
        "",
        "## Phase 2 — 结构资格与冻结任务 roster",
        "",
        "结构筛选不读取 intervention outcome 或 proposed-method gain。六个任务都能定义稳定 phase anchor、environment-only 注入、注入瞬间无直接 cause violation、chunk 内至少剩余 8 个动作，并能进行 fresh-env prefix reconstruction；未替换任务。",
        "",
        "| Task | Family | Structural status | Smoke config | Demo |",
        "| --- | --- | --- | --- | ---: |",
    ]
    eligibility = context["eligibility"]
    for record in eligibility:
        lines.append(
            f"| {TASK_LABELS[record['task']]} | {record['family']} | {record['status']} | "
            f"{record['structural_smoke_config_id']} | {record['structural_smoke_demo_id']} |"
        )
    lines.extend(
        [
            "",
            f"结果：`{roster['eligible_task_count']}/6` structurally eligible，replacement_count=`{roster['replacement_count']}`。",
            "",
            "## Phase 3 — 固定扰动网格资格",
            "",
            "只使用 calibration demo 0–9；evaluation 10–39 与 held-out 40–49 均未检查。固定门槛为 immediate violation ≤10%、delayed nominal violation 30–80%、fresh replay ≥99%，并要求十个 calibration 事件 phase-valid。总计 88 个配置、880 条 calibration nominal 记录。",
            "",
            "| Task | Qualified configs | Frozen primary | Immediate | Delayed nominal | Replay |",
            "| --- | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for task in TASK_NAMES:
        item = perturbations["tasks"][task]
        lines.append(
            f"| {TASK_LABELS[task]} | {item['qualified_config_count']} | "
            f"{item['selected_config_id'] or 'none'} | "
            f"{fmt(item['selected_immediate_violation_rate'])} | "
            f"{fmt(item['selected_delayed_violation_rate'])} | "
            f"{fmt(item['selected_replay_rate'])} |"
        )
    lines.extend(
        [
            "",
            "通过的两任务为 cream-cheese → bowl（10 个配置，primary=`lead08_shift040mm`）和 bowl → stove（2 个配置，primary=`lead14_lateral050mm`）。另外四任务是冻结负结果：bowl/plate 在 immediate contact 与 0/100% delayed rate 间跳变；两个 drawer 任务 delayed rate 最高仅 20%/10%；push 任务的结构注入合法但 16-step suffix 内 delayed rate 为 0。未在 evaluation split 调参。",
            "",
            "为了满足未来‘至少两种 severity’而不看方法收益，preflight 从已合格 calibration 配置中机械冻结第二种 severity：cream 使用 `lead10_shift060mm`，bowl/stove 使用 `lead14_lateral075mm`。每个任务两种 severity 各 25 events。",
            "",
            "## Phase 4 — Strong small actor",
            "",
            "固定 actor 为 shared multitask `HistoryConditionedStateACT`：4 帧 95-D state history、3 个已执行动作、task ID、16 个 action queries；d_model=192，6 heads，3 encoder + 3 decoder layers，FFN=768，dropout=0.10。模型参数 `3,146,503`，低于 10M；无 RGB、privileged perturbation metadata 或 world model。每任务 50 demos，seeds 7/17/29，6000 steps，batch 256，AdamW lr=3e-4 / wd=1e-4，execution horizon=1，无 sweep。",
            "",
            "2 tasks × 2 demos × 32 states、3 steps 的 GPU smoke 成功，完整 optimizer/RNG checkpoint 与 portable checkpoint 都能写入。正式训练结果：",
            "",
            "| Seed | Final loss | Position | Gripper | Step |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for seed in (7, 17, 29):
        item = training[str(seed)]
        lines.append(
            f"| {seed} | {item['loss']:.5f} | {item['position_loss']:.5f} | "
            f"{item['gripper_loss']:.5f} | {item['step']} |"
        )
    lines.extend(
        [
            "",
            "### Clean success / phase reach gate",
            "",
            "每任务每 seed 10 个相同 clean init states，共 180 episodes。A 门为 clean success ∈[0.30,0.90]；B 门为 late phase reach ≥0.70 且失败主要发生在到达该 phase 后。A 或 B 任一满足即 task qualified。",
            "",
            "| Task | Seed success 7/17/29 | Clean | Grasp/open | Lift/transport | Late phase | Qualified |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for task in TASK_NAMES:
        item = actor["tasks"][task]
        seed_rates = "/".join(
            f"{item['per_seed'][str(seed)]['success_rate']:.1f}"
            for seed in (7, 17, 29)
        )
        lines.append(
            f"| {TASK_LABELS[task]} | {seed_rates} | {item['clean_success_rate']:.3f} | "
            f"{item['grasp_or_open_phase_reach_rate']:.3f} | "
            f"{item['lift_or_transport_phase_reach_rate']:.3f} | "
            f"{item['pre_release_or_contact_phase_reach_rate']:.3f} | "
            f"{item['qualification_status']} |"
        )
    lines.extend(
        [
            "",
            f"结果：`{actor['qualified_task_count']}/6` actor-qualified，**actor gate=`{actor['actor_gate']}`**。与扰动资格的交集恰好是 cream-cheese → bowl 和 bowl → stove，两任务均 actor-qualified。push 虽 aggregate=0.40，但 seed success=0.9/0.0/0.3，后续不能把它解释成稳定 actor substrate；它同时也没有合格扰动。",
            "",
            "## Actor 机制反解（post-hoc、non-gate）",
            "",
            "弱 reference `ChunkedBCMLP` 只看当前 95-D state、按任务独立训练，并用三层 GELU MLP 直接回归 16-action chunk；新 ACT 增加已执行动作历史、state history、task token 和 transformer query decoder，同时改为 shared multitask。两项严格 matched 的 Stage-1 horizon-1 clean 比较只能识别整个 architecture/training bundle，不能识别单一组件因果：",
            "",
            "- bowl/plate：0.067 → 0.333，+0.267；三 seed 范围仅 0.10，但主要瓶颈仍是 lift/transport（reach 0.333）。",
            "- middle drawer：0.033 → 0.233，+0.200；仍未过 gate，主要瓶颈是早期 grasp/articulation entry（reach 0.300）。",
            "",
            "在冻结 checkpoint 的 teacher-forced cache rows 上逐一遮蔽输入，不重训、不参与 gate。相对 full actor 的首动作 position MAE 变化如下：",
            "",
            "| Task | Repeat-current state | Zero action history | Cyclic task ID | Seed success range |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for task in TASK_NAMES:
        item = mechanism["tasks"][task]
        delta = item["first_action_position_mae_delta_percent"]
        lines.append(
            f"| {TASK_LABELS[task]} | {delta['repeat_current_state']:+.2f}% | "
            f"{delta['zero_action_history']:+.2f}% | {delta['cyclic_task_id']:+.2f}% | "
            f"{item['act_seed_success_range']:.3f} |"
        )
    lines.extend(
        [
            "",
            "六任务最大的 offline dependence 都是 executed-action history（+548% 到 +962% MAE），task ID 次之（+52% 到 +111%），简单压平 state history 的影响较小（+3% 到 +10%）。结合代码路径，这支持一个受限解释：动作历史为 shared actor 提供了当前 state 难以唯一表达的执行上下文，可能解释 matched tasks 的部分提升；task token 防止多任务动作模式混叠。它不证明因果，因为遮蔽是分布外诊断，且 ACT 与 weak MLP 同时改变了架构、共享方式和训练步数。",
            "",
            "降低/不稳定机制也能定位但不能过度归因：两个 drawer 任务在进入早期操作 phase 前就大量失败；push 的 0.90 seed range 与对 action-history/task-ID 的极强 offline sensitivity 一致，说明其 aggregate 提升由 seed-specific trajectory mode 主导，不是稳健机制；bowl/stove 0.90 接近 gate 上界，但没有 matched weak reference，不能声称具体组件带来提升。没有生成新 idea，也没有训练 Track-B diagnostic probe。",
            "",
            "## Phase 5 — Fresh-environment replay gate",
            "",
            "正锚为合格 cream-cheese 配置，负锚为无合格配置的 middle drawer structural smoke。每个锚在 k=2/8/16 各用 3 个独立 fresh environment 重建；不共享 mutable env，snapshot restore 不是 primary initializer。",
            "",
            "| Anchor | Task | Branch groups | Pass | State error | Contact/outcome | Chunk hash |",
            "| --- | --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for label in ("positive", "negative"):
        item = replay["anchors"][label]
        lines.append(
            f"| {label} | {TASK_LABELS[item['task']]} | {item['branch_point_count']} | "
            f"{item['pass_count']}/{item['branch_point_count']} | {item['max_state_error']} | "
            f"{item['contact_outcome_agreement']} | {item['action_chunk_hash_exact']} |"
        )
    lines.extend(
        [
            "",
            f"总体：state match `{replay['branch_point_pass_count']}/{replay['branch_point_count']}`，max error `{replay['max_state_error']}`，**replay gate=`{replay['status']}`**。",
            "",
            "## Novelty 双审边界",
            "",
            "两名 reviewer 独立收敛：广义 policy-relative recovery/replanability window 已被 CheckVLA、prefix-Q chunking、DEHP/BCP 和 option interruption 强覆盖，不能赋 N3。仅保留窄的 oracle protocol 边界：检测后显式 target-cause S(k) 过滤、冻结 actor 的 Cπ(k)、真实 stride-1 分支，以及 interior optimum 相对 d/k_last_safe 的比较。当前标签为 **`N2_ORACLE_PROTOCOL_BOUNDARY_ONLY`**；在完整 Atlas 出现可复现 interior oracle gap 前没有算法新颖性。",
            "",
            "## Atlas 预估工作量（尚未执行）",
            "",
            f"冻结交集为 `{preflight['eligible_task_count']}` tasks × 每任务 50 events（30 evaluation + 20 held-out）× 3 actor seeds × 14 个 stride-1 prefixes（k=2…15），因此 primary R(k) 为 `{preflight['primary_R_branch_run_count']}` branch runs。三种 secondary operators 只在 d/k_best/k_last_safe 运行，最多 `{preflight['secondary_operator_branch_run_count_max']}`，总上限 `{preflight['estimated_total_branch_run_count_max']}`；不含 source-prefix reconstruction overhead。B0–B5、A_original、N_oracle 从相同 R(k) branches 派生，不重复乘七。",
            "",
            "完整 frozen event schedule 已生成，但状态明确为 `PREFLIGHT_ONLY_NOT_LAUNCHED`。",
            "",
            "## 资源、恢复与测试",
            "",
            "- 在开发机 PAI DSW 本地完成；机器有 4×A800，但本轮最多只有 1 个 GPU worker（低于计划上限 2），CPU 校准与 replay 可并行。",
            "- 未提交 PAI DLC/spot/web-orchestrator 外部作业：本地 A800 足以完成首 checkpoint，遵守 local-first 与不做无意义 probe 的约束。",
            "- 正式 actor 每 2000 steps 保存 model/optimizer/scheduler/RNG 完整 checkpoint，启动时扫描 complete marker 自动恢复；portable model 权重进入 Git。",
            f"- 单元/契约测试：`{context['tests']['passed']} passed`，`{context['tests']['warnings']} warnings`；测试 XML 已归档。",
            "- 两个长进程（扰动资格、seed29 eval）在运行期间外层 `commands.sh` 被编辑，模块已原子写完后旧 Bash 读取偏移文本导致尾部 exit 127。已按 schema、行数（880 calibration；60 seed29 episodes）和 summary 验证数据完整；该 wrapper incident 单独归档，未伪装成 clean exit。",
            "",
            "## 当前 blocker 与下一条命令",
            "",
            "没有 actor、replay 或基础设施 blocker。当前唯一停止条件是计划规定的 first-checkpoint barrier；同时后续范围被四个扰动负任务收窄到两任务。完整 Atlas runner 尚未运行，Track A/Track B 的统计结论仍不可得。",
            "",
            "下一条安全、可复现命令是（重验 frozen intersection、severity/event schedule 和 6,900 branch 上限，不执行任何 branch）：",
            "",
            "```bash",
            NEXT_COMMAND,
            "```",
            "",
            "本 checkpoint 报告写入并交付后，下一阶段才能在该 frozen schedule 上实现并先运行单 event/单 seed CPU branch smoke；在 smoke 的 replay/hash/预算字段通过前，不应提交完整 Atlas。",
            "",
            "## 决策字段的严格分离",
            "",
            "- `stage1b_universal_hypothesis`: **KILLED_IMMUTABLE**。",
            "- `track_a_conditional_original`: **INCONCLUSIVE**（Atlas 未运行）。",
            "- `track_b_replanability`: **INCONCLUSIVE**（Atlas 未运行）。",
            "- diagnostic predictability: **NOT_RUN**；本轮 input masking 只是 actor 机理反解，不是 Track-B probe。",
            "- `operator_router`: **NOT_ACTIVATED**。",
            "- learned/deployable evidence: **none**。",
            "- `overall`: 未赋值；最终枚举必须等待 Atlas，不能用 checkpoint PASS 冒充最终研究结论。",
            "- `accepted=false`；full_atlas_launched=false。",
            "",
            "## 关键证据路径",
            "",
            "- `experiments/r16_p14_stage2a/preregistration.yaml`：冻结合同和两次 pre-calibration structural amendments。",
            "- `artifacts/stage2a/perturbations/`：880 raw records、88-grid summary、negative results、frozen parameters。",
            "- `artifacts/stage2a/actor/`：3 portable checkpoints、180 rollout records、qualification 与 mechanism audit。",
            "- `artifacts/stage2a/replay/`：18 fresh-env reconstructions 与 gate summary。",
            "- `artifacts/stage2a/atlas/`：仅 preflight schedule，未有 branch output。",
            "- `artifacts/stage2a/SHA256SUMS`：首 checkpoint artifact hashes。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    experiment = repo / "experiments/r16_p14_stage2a"
    artifact = repo / "artifacts/stage2a"
    actor = load(artifact / "actor/qualification_summary.json")
    context: dict[str, Any] = {
        "head": git_value(repo, "rev-parse", "HEAD"),
        "tree": git_value(repo, "rev-parse", "HEAD^{tree}"),
        "branch": git_value(repo, "branch", "--show-current"),
        "remote": git_value(repo, "remote", "get-url", "origin"),
        "source": load(artifact / "source_freeze/source_manifest.json"),
        "clue": load(experiment / "reports/current_clue_reproduction.json"),
        "roster": load(experiment / "task_screen/task_roster_frozen.json"),
        "eligibility": [
            json.loads(line)
            for line in (experiment / "task_screen/eligibility.jsonl").read_text().splitlines()
            if line.strip()
        ],
        "perturbations": load(artifact / "perturbations/frozen_parameters.json"),
        "actor": actor,
        "training": {
            str(seed): load(artifact / f"actor/seed_{seed}.training_summary.json")
            for seed in (7, 17, 29)
        },
        "mechanism": load(artifact / "actor/mechanism_audit/summary.json"),
        "replay": load(artifact / "replay/summary.json"),
        "preflight": load(artifact / "atlas/preflight_summary.json"),
        "tests": load(artifact / "test_results/summary.json"),
    }
    summary = {
        "schema_version": 1,
        "checkpoint_status": "FIRST_CHECKPOINT_PASS_READY_FOR_BOUNDED_ATLAS",
        "evidence_commit": context["head"],
        "evidence_tree": context["tree"],
        "stage1b_universal_hypothesis": "KILLED_IMMUTABLE",
        "actor_gate": actor["actor_gate"],
        "actor_qualified_task_count": actor["qualified_task_count"],
        "perturbation_qualified_task_count": context["perturbations"]["qualified_task_count"],
        "atlas_eligible_task_count": context["preflight"]["eligible_task_count"],
        "replay_gate": context["replay"]["status"],
        "estimated_primary_R_branches": context["preflight"]["primary_R_branch_run_count"],
        "estimated_total_branches_max": context["preflight"]["estimated_total_branch_run_count_max"],
        "current_blocker": "CHECKPOINT_BARRIER_ONLY",
        "exact_next_command": NEXT_COMMAND,
        "full_atlas_launched": False,
        "track_a_conditional_original": "INCONCLUSIVE",
        "track_b_replanability": "INCONCLUSIVE",
        "operator_router": "NOT_ACTIVATED",
        "accepted": False,
        "report_sha256": None,
    }
    report = render(context)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "REPORT.md"
    atomic_write_text(report_path, report)
    summary["report_sha256"] = sha256(report_path)
    atomic_write_json(args.output_dir / "first_checkpoint_summary.json", summary)
    decision = {
        "schema_version": 1,
        "stage": "R16-P14 Stage-2A",
        "checkpoint": "FIRST_CHECKPOINT_COMPLETE",
        "checkpoint_status": "PASS_READY_FOR_BOUNDED_ATLAS",
        "evidence_commit": context["head"],
        "evidence_tree": context["tree"],
        "stage1b_universal_hypothesis": "KILLED_IMMUTABLE",
        "track_a_conditional_original": "INCONCLUSIVE",
        "track_b_replanability": "INCONCLUSIVE",
        "operator_router": "NOT_ACTIVATED",
        "overall": None,
        "overall_assignment_status": "DEFERRED_UNTIL_FULL_ATLAS",
        "actor_gate": actor["actor_gate"],
        "replay_gate": context["replay"]["status"],
        "perturbation_qualified_task_count": context["perturbations"]["qualified_task_count"],
        "atlas_eligible_task_count": context["preflight"]["eligible_task_count"],
        "accepted": False,
        "full_atlas_launched": False,
        "diagnostic_probe_trained": False,
        "posthoc_actor_mechanism_audit_is_not_track_b_probe": True,
    }
    atomic_write_json(experiment / "decision.json", decision)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
