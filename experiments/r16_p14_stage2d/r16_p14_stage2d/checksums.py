from __future__ import annotations

from pathlib import Path

from .io_utils import atomic_write_text, sha256_file
from .settings import ARTIFACT_ROOT


CANONICAL_ARTIFACT_ROOT = Path("artifacts/stage2d")


def canonical_artifact_path(path: Path) -> Path:
    """Serialize one stable repository path for local and external PAI roots."""
    return CANONICAL_ARTIFACT_ROOT / path.relative_to(ARTIFACT_ROOT)


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
