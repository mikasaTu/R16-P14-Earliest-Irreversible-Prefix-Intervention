from __future__ import annotations

import json

from .io_utils import atomic_write_json, atomic_write_text
from .settings import ARTIFACT_ROOT, EXPERIMENT_ROOT


def main() -> None:
    phase_a = json.loads((ARTIFACT_ROOT / "chunk_executability/summary.json").read_text())
    events = json.loads((ARTIFACT_ROOT / "actor_events/summary.json").read_text())
    phase_c = json.loads((ARTIFACT_ROOT / "perturbation_qualification/summary.json").read_text())
    phase_d = json.loads((ARTIFACT_ROOT / "replay/summary.json").read_text())
    gates = {
        "H_valid_ge_8": phase_a["H_valid"] is not None and int(phase_a["H_valid"]) >= 8,
        "both_tasks_two_qualifying_severities": phase_c["both_tasks_have_two_qualifying_severities"],
        "actor_history_replay_pass": phase_d["status"] == "PASS",
        "event_split_pass": not events["event_split_blocked"],
    }
    payload = {
        "schema_version": 1,
        "phase_a_status": phase_a["status"],
        "H_valid": phase_a["H_valid"],
        "phase_c_status": phase_c["status"],
        "phase_d_status": phase_d["status"],
        "gates": gates,
        "automatic_continuation_contract_passed": all(gates.values()),
        "formal_positive_labels_permitted": all(gates.values()),
        "phase_e_f_execution_required_by_latest_user_instruction": True,
        "early_stop_applied": False,
    }
    report = "\n".join(
        [
            "# Stage-2B A-D checkpoint",
            "",
            f"A/C/D joint gate: **{'PASS' if all(gates.values()) else 'BLOCKED'}**; `H_valid={phase_a['H_valid']}`.",
            "",
            f"- Chunk executability: `{phase_a['status']}`",
            f"- Event split: `{'PASS' if not events['event_split_blocked'] else 'BLOCKED'}`",
            f"- Actor-conditioned perturbation: `{phase_c['status']}`",
            f"- Full actor-history replay: `{phase_d['status']}`",
            "",
            "The latest user instruction requires Phase E/F to execute even when this checkpoint is blocked. In that case all downstream results are descriptive and cannot receive a positive Track label.",
            "",
        ]
    )
    atomic_write_json(ARTIFACT_ROOT / "checkpoint/summary.json", payload)
    atomic_write_text(ARTIFACT_ROOT / "checkpoint/REPORT.md", report)
    atomic_write_json(EXPERIMENT_ROOT / "reports/checkpoint_summary.json", payload)
    atomic_write_text(EXPERIMENT_ROOT / "reports/checkpoint_report.md", report)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
