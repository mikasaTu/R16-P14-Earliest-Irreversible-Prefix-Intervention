from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from .io_utils import atomic_write_json, atomic_write_text
from .settings import DEFAULT_LIBERO_CONFIG, PROJECT_ROOT, TASK_SPECS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=PROJECT_ROOT, text=True
    ).strip()


def hash_named_files(paths: Iterable[Path]) -> tuple[str, list[dict[str, object]]]:
    digest = hashlib.sha256()
    records: list[dict[str, object]] = []
    for path in sorted(paths):
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        value = sha256_file(path)
        size = path.stat().st_size
        digest.update(relative.encode("utf-8") + b"\0" + value.encode("ascii") + b"\n")
        records.append({"path": relative, "size_bytes": size, "sha256": value})
    return digest.hexdigest(), records


def package_version(name: str) -> str | None:
    try:
        module = __import__(name)
    except Exception:
        return None
    return str(getattr(module, "__version__", "unknown"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    tracked = [PROJECT_ROOT / path for path in git("ls-files", "libero").splitlines()]
    source_hash, source_files = hash_named_files(path for path in tracked if path.is_file())
    source_lines = "".join(
        f"{record['sha256']}  {record['path']}\n" for record in source_files
    )
    atomic_write_text(output_dir / "libero_source_files.sha256", source_lines)
    immutable = {
        "stage1_report": PROJECT_ROOT / "experiments/r16_p14_libero_stage1/reports/REPORT.md",
        "stage1_decision": PROJECT_ROOT / "experiments/r16_p14_libero_stage1/decision.json",
        "stage1b_report": PROJECT_ROOT / "experiments/r16_p14_libero_stage1b/reports/REPORT.md",
        "stage1b_decision": PROJECT_ROOT / "experiments/r16_p14_libero_stage1b/decision.json",
        "stage1b_selected_metrics": PROJECT_ROOT / "artifacts/stage1b/expert_chunk_calibration/selected_config_paired_metrics.csv",
    }
    demos = {
        spec.hdf5_name: {
            "path": str(spec.hdf5_path),
            "size_bytes": spec.hdf5_path.stat().st_size,
            "sha256": sha256_file(spec.hdf5_path),
        }
        for spec in TASK_SPECS.values()
    }
    result = {
        "schema_version": 1,
        "repository": {
            "head": git("rev-parse", "HEAD"),
            "head_tree": git("rev-parse", "HEAD^{tree}"),
            "branch": git("branch", "--show-current"),
            "remote": git("remote", "get-url", "origin"),
            "working_tree_status_at_freeze": git("status", "--short"),
        },
        "stage1_frozen": {
            "commit": "a1b61194a8382f5b1a247b9cd9b140645ff2aeb8",
            "tree": "53001c43fbbb165c0a1f2c71f9cbd4c81b9d0ced",
        },
        "stage1b_frozen": {
            "commit": "e29e3ead42fd1799b412a4968e6a67aac3784874",
            "tree": "4016b05942e6dfb291f1bc3a2644e177a208b608",
            "decision": "KILL_CORE_HYPOTHESIS",
        },
        "immutable_file_sha256": {
            name: sha256_file(path) for name, path in immutable.items() if path.is_file()
        },
        "libero": {
            "tracked_source_combined_sha256": source_hash,
            "tracked_source_file_count": len(source_files),
            "config_path": str(DEFAULT_LIBERO_CONFIG / "config.yaml"),
            "config_sha256": sha256_file(DEFAULT_LIBERO_CONFIG / "config.yaml"),
        },
        "demonstrations": demos,
        "runtime": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "torch": package_version("torch"),
            "torch_cuda": (
                str(__import__("torch").version.cuda)
                if package_version("torch") is not None
                else None
            ),
            "mujoco": package_version("mujoco"),
            "robosuite": package_version("robosuite"),
            "numpy": package_version("numpy"),
            "h5py": package_version("h5py"),
        },
    }
    atomic_write_json(output_dir / "source_manifest.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
