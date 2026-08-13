from __future__ import annotations

from pathlib import Path

from r16_p14_stage1b.release import manifest_files


def test_manifest_excludes_itself_and_python_cache(tmp_path: Path) -> None:
    experiment = tmp_path / "experiments/r16_p14_libero_stage1b"
    artifacts = tmp_path / "artifacts/stage1b"
    (experiment / "__pycache__").mkdir(parents=True)
    artifacts.mkdir(parents=True)
    (experiment / "source.py").write_text("ok")
    (experiment / "__pycache__/source.pyc").write_bytes(b"cache")
    (artifacts / "result.json").write_text("{}")
    (artifacts / "SHA256SUMS").write_text("old")
    relatives = [path.relative_to(tmp_path) for path in manifest_files(tmp_path)]
    assert relatives == [
        Path("artifacts/stage1b/result.json"),
        Path("experiments/r16_p14_libero_stage1b/source.py"),
    ]
