from __future__ import annotations

from pathlib import Path

from .io_utils import atomic_write_text, sha256_file
from .settings import ARTIFACT_ROOT


CANONICAL_ARTIFACT_ROOT = Path("artifacts/stage2d")


def canonical_artifact_path(path: Path) -> Path:
    """Serialize one stable repository path for local and external PAI roots."""
    try:
        relative = path.relative_to(ARTIFACT_ROOT)
    except ValueError:
        # PAI writes to `<run>/stage2d_artifacts` while the imported evidence
        # lives under `artifacts/stage2d`.  Both must produce the same
        # repository-relative checksum key; never serialize a machine-local
        # absolute path into SHA256SUMS.
        parts = path.resolve().parts
        try:
            marker = parts.index("stage2d_artifacts")
        except ValueError as exc:
            raise ValueError(f"not a Stage-2D artifact path: {path}") from exc
        relative = Path(*parts[marker + 1 :])
    return CANONICAL_ARTIFACT_ROOT / relative


def main() -> None:
    target = ARTIFACT_ROOT / "SHA256SUMS"
    files = [
        path
        for path in sorted(ARTIFACT_ROOT.rglob("*"))
        if path.is_file() and path != target
    ]
    lines = [
        f"{sha256_file(path)}  {canonical_artifact_path(path)}"
        for path in files
    ]
    atomic_write_text(target, "\n".join(lines) + "\n")
    print(f"STAGE2D_CHECKSUMS files={len(files)} path={target}")


if __name__ == "__main__":
    main()
