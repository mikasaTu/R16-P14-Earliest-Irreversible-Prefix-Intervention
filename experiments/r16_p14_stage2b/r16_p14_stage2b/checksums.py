from __future__ import annotations

import json
from pathlib import Path

from .io_utils import atomic_write_json, atomic_write_text, sha256_file
from .settings import ARTIFACT_ROOT, EXPERIMENT_ROOT, PROJECT_ROOT


def main() -> None:
    targets = []
    for root in (ARTIFACT_ROOT, EXPERIMENT_ROOT):
        for path in root.rglob("*"):
            if not path.is_file() or path.name in {"SHA256SUMS", "checksum_manifest.json"}:
                continue
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            targets.append(path)
    records = []
    lines = []
    for path in sorted(set(targets)):
        relative = path.relative_to(PROJECT_ROOT)
        digest = sha256_file(path)
        records.append({"path": str(relative), "size_bytes": path.stat().st_size, "sha256": digest})
        lines.append(f"{digest}  {relative}\n")
    atomic_write_text(ARTIFACT_ROOT / "SHA256SUMS", "".join(lines))
    manifest = {"schema_version": 1, "algorithm": "SHA256", "file_count": len(records), "files": records}
    atomic_write_json(ARTIFACT_ROOT / "checksum_manifest.json", manifest)
    atomic_write_json(EXPERIMENT_ROOT / "reports/checksum_manifest.json", manifest)
    print(json.dumps({"file_count": len(records), "sha256sums": str(ARTIFACT_ROOT / 'SHA256SUMS')}, indent=2))


if __name__ == "__main__":
    main()
