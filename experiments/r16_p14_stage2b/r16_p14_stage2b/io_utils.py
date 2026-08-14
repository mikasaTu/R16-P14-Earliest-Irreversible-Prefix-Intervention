from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: Any, dtype: Any | None = None) -> str:
    array = np.asarray(value, dtype=dtype)
    return sha256_bytes(np.ascontiguousarray(array).tobytes())


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def _atomic_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp.{os.getpid()}")


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _atomic_path(path)
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def atomic_write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    text = "".join(canonical_json(record) + "\n" for record in records)
    atomic_write_text(path, text)


def write_once_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    materialized = list(records)
    if path.exists():
        existing = load_jsonl(path)
        if existing != materialized:
            raise FileExistsError(f"non-empty completed evidence differs: {path}")
        return
    atomic_write_jsonl(path, materialized)


def atomic_write_csv(path: Path, fieldnames: Sequence[str], records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _atomic_path(path)
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_once_json(path: Path, value: Any) -> None:
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != value:
            raise FileExistsError(f"non-empty completed evidence differs: {path}")
        return
    atomic_write_json(path, value)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def merge_json_files(directory: Path, output: Path) -> list[dict[str, Any]]:
    records = [json.loads(path.read_text()) for path in sorted(directory.glob("*.json"))]
    atomic_write_jsonl(output, records)
    return records


def merge_jsonl_files(paths: Iterable[Path], output: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        records.extend(load_jsonl(path))
    atomic_write_jsonl(output, records)
    return records
