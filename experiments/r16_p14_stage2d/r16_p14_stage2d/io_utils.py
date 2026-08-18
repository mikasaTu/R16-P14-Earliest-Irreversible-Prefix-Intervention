from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def sha256_array(value: Any, dtype: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    header = json.dumps(
        {"dtype": str(array.dtype), "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(header + b"\0" + array.tobytes()).hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_write_text(
        path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL, transparently assembling a size-limited shard set.

    GitHub rejects individual files larger than 100 MB.  Large immutable
    evidence files therefore use a small JSONL pointer at ``path`` and
    newline-preserving shards in ``<path>.parts``.  The pointer retains the
    original byte hash and row count; callers continue to use the canonical
    logical path and receive the same ordered rows.
    """
    if path.is_file():
        with path.open() as handle:
            lines = [line for line in handle if line.strip()]
        if len(lines) == 1:
            candidate = json.loads(lines[0])
            if isinstance(candidate, dict) and candidate.get("_sharded_jsonl") is True:
                parts_dir = path.parent / str(candidate["parts_dir"])
                rows = []
                for shard in sorted(parts_dir.glob("*.jsonl")):
                    with shard.open() as handle:
                        rows.extend(json.loads(line) for line in handle if line.strip())
                expected_rows = candidate.get("row_count")
                if expected_rows is not None and len(rows) != int(expected_rows):
                    raise ValueError(
                        f"sharded JSONL row count mismatch for {path}: "
                        f"{len(rows)} != {expected_rows}"
                    )
                return rows
        return [json.loads(line) for line in lines]
    parts_dir = path.parent / f"{path.name}.parts"
    if parts_dir.is_dir():
        rows = []
        for shard in sorted(parts_dir.glob("*.jsonl")):
            with shard.open() as handle:
                rows.extend(json.loads(line) for line in handle if line.strip())
        return rows
    raise FileNotFoundError(path)
