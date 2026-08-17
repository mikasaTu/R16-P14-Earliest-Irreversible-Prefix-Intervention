from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .fresh_process import run_spawned_branch
from .io_utils import atomic_write_json
from .isolation import select_smoke_events, smoke_parameter
from .settings import ARTIFACT_ROOT, TASKS


def main() -> None:
    events = select_smoke_events()
    rows: list[dict[str, Any]] = []
    for task in TASKS:
        row = run_spawned_branch(
            event=events[task],
            parameter=smoke_parameter(events[task]),
            prefix_k=6,
            arm="CACHED_MATCHED",
            repeat=0,
            device="cpu",
        )
        rows.append(row)
        print(
            f"REAL_LIBERO_SMOKE task={task} error={int(bool(row.get('error')))} "
            f"pid={row.get('pid')}",
            flush=True,
        )
    passed = all(
        not row.get("error")
        and row.get("unique_process_contract")
        and row.get("reconstruction", {}).get("actor_inference_side_effect_free")
        and row.get("tail_horizon") == 4
        for row in rows
    )
    payload = {
        "schema_version": 1,
        "status": "PASS" if passed else "FAIL",
        "real_libero_tasks": list(TASKS),
        "local_before_pai": True,
        "rows": rows,
    }
    atomic_write_json(ARTIFACT_ROOT / "test_results/integration_smoke.json", payload)
    print(json.dumps({"status": payload["status"], "tasks": list(TASKS)}, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
