from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from r16_p14_stage2b.io_utils import atomic_write_text

from .settings import ARTIFACT_ROOT


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(root: Path, output_name: str = "SHA256SUMS") -> str:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root)
        if relative.as_posix() == output_name:
            continue
        rows.append(f"{sha256_file(path)}  {relative.as_posix()}")
    return "\n".join(rows) + ("\n" if rows else "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--output-name", default="SHA256SUMS")
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / args.output_name
    atomic_write_text(output, build_manifest(root, args.output_name))
    print(f"STAGE2C_SHA256_COMPLETE files={len(output.read_text().splitlines())} output={output}")


if __name__ == "__main__":
    main()
