"""Publish compact, provenance-linked projections of immutable S1 row files.

The source consolidation remains the scientific source of truth.  This utility
only makes a Git-friendly projection: every input JSONL byte is retained in a
deterministic gzip member, while the checked-in JSONL drops the largest raw
payload fields and records the source shard used for provenance.  It never
reads evaluation input before the canonical diagnostic admission call.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:  # repository invocation with experiments/r16_p14_stage2f on PYTHONPATH
    from s1 import matrix  # type: ignore
except ImportError:  # pragma: no cover - direct source invocation fallback
    try:
        import matrix  # type: ignore
    except ImportError:  # phase-2 admission reports a closed gate below
        matrix = None  # type: ignore

MAX_OUTPUT_BYTES = 95 * 1024 * 1024
REMOVED_FIELDS = (
    "d4_signatures",
    "labels",
    "topology_candidate",
    "recovery_action_hashes",
    "recovery_chunk_hashes",
)
REMOVED_FIELDS_SET = frozenset(REMOVED_FIELDS)
ACTION_STREAM_DERIVATION = "sha256(canonical_json(recovery_action_hashes))"
IDENTITY_FIELDS = (
    "task",
    "event_instance_id",
    "init_state_id",
    "split",
    "generator_actor_seed",
    "recovery_actor_seed",
    "operator",
    "prefix_k",
    "tail_horizon",
    "action_budget",
    "policy_call_cap",
)
STRING_IDENTITY_FIELDS = frozenset({"task", "event_instance_id", "init_state_id", "split", "operator"})
INTEGER_IDENTITY_FIELDS = frozenset(
    {
        "generator_actor_seed",
        "recovery_actor_seed",
        "prefix_k",
        "tail_horizon",
        "action_budget",
        "policy_call_cap",
    }
)
TASKS = frozenset(
    {
        "put_the_cream_cheese_in_the_bowl",
        "put_the_bowl_on_the_plate",
    }
)
ROW_FILES = {
    "phase1": ("grid_rows.jsonl", "reference_rows.jsonl"),
    "phase2": ("atlas_rows.jsonl", "reference_rows.jsonl"),
}
RECEIPT_NAME = "project_published_rows_receipt.json"


class PublicationError(RuntimeError):
    """Raised internally when a publication precondition fails closed."""


def _canonical_json(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise PublicationError(f"value is not canonical JSON: {exc}") from exc
    return text.encode("utf-8")


def _json_line(value: Mapping[str, Any]) -> bytes:
    return _canonical_json(dict(value)) + b"\n"


def _sha256_stream(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
                total += len(block)
    except OSError as exc:
        raise PublicationError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest(), total


def _sha256_gzip_uncompressed(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    try:
        with gzip.open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
                total += len(block)
    except (OSError, EOFError, gzip.BadGzipFile) as exc:
        raise PublicationError(f"cannot verify gzip {path}: {exc}") from exc
    return digest.hexdigest(), total


def _strict_equal(left: Any, right: Any) -> bool:
    """JSON equality that does not conflate ``true`` with ``1``."""
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        if set(left) != set(right):
            return False
        return all(_strict_equal(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _strict_equal(lhs, rhs) for lhs, rhs in zip(left, right)
        )
    return left == right


def _identity_key(row: Mapping[str, Any], label: str) -> tuple[Any, ...]:
    values: list[Any] = []
    for field in IDENTITY_FIELDS:
        if field not in row:
            raise PublicationError(f"{label}: missing branch identity field {field}")
        value = row[field]
        if field in STRING_IDENTITY_FIELDS:
            if not isinstance(value, str) or not value:
                raise PublicationError(f"{label}: {field} must be a non-empty string")
        elif field in INTEGER_IDENTITY_FIELDS:
            if not isinstance(value, int) or isinstance(value, bool):
                raise PublicationError(f"{label}: {field} must be an integer")
        try:
            hash(value)
        except TypeError as exc:
            raise PublicationError(f"{label}: identity field {field} is unhashable") from exc
        values.append(value)
    if row["task"] not in TASKS:
        raise PublicationError(f"{label}: unknown task {row['task']!r}")
    return tuple(values)


def _validate_status(row: Mapping[str, Any], label: str) -> None:
    status = row.get("status")
    if status == "COMPLETE":
        return
    if (
        isinstance(status, str)
        and status.strip().upper().startswith("BLOCKED")
        and row.get("error_type") == "PrefixOutsideTaskHorizon"
    ):
        return
    raise PublicationError(
        f"{label}: unknown/error row cannot be published as science data (status={status!r})"
    )


def _raw_shard_path(source_root: Path, phase: str, row: Mapping[str, Any], label: str) -> tuple[Path, str]:
    task = row.get("task")
    trace_path = row.get("trace_path")
    if not isinstance(trace_path, str) or not trace_path.strip():
        raise PublicationError(f"{label}: trace_path is required for raw shard provenance")
    basename = Path(trace_path).name
    suffix = ".jsonl.gz"
    if not basename.endswith(suffix) or basename == suffix:
        raise PublicationError(f"{label}: trace_path must name a .jsonl.gz trace")
    shard_name = basename[: -len(suffix)] + ".json"
    task_dir = (source_root / phase / "shards" / str(task)).resolve()
    raw_path = (task_dir / shard_name).resolve()
    if raw_path.parent != task_dir:
        raise PublicationError(f"{label}: derived raw shard escapes task directory")
    if not raw_path.is_file():
        raise PublicationError(f"{label}: missing raw shard {raw_path}")
    # This is deliberately derived from the current phase.  A Phase-2 row may
    # reuse a Phase-1 trace, but its source row is phase2/shards/<task>/<name>.
    relative = Path(phase) / "shards" / str(task) / shard_name
    return raw_path, relative.as_posix()


def _read_raw_shard(path: Path, row: Mapping[str, Any], label: str) -> str:
    raw_sha, _ = _sha256_stream(path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            source = json.load(handle)
    except (OSError, UnicodeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise PublicationError(f"{label}: invalid raw shard {path}: {exc}") from exc
    if not isinstance(source, Mapping):
        raise PublicationError(f"{label}: raw shard {path} is not a JSON object")
    for field in IDENTITY_FIELDS:
        if field not in source or not _strict_equal(source[field], row[field]):
            raise PublicationError(f"{label}: raw shard {path} disagrees on {field}")
    # safe_success is the scientific outcome and is checked whenever the
    # consolidated row carries it.  Structural rows may legitimately omit it.
    if "safe_success" in row:
        if "safe_success" not in source or not _strict_equal(source["safe_success"], row["safe_success"]):
            raise PublicationError(f"{label}: raw shard {path} disagrees on safe_success")
    return raw_sha


def _compact_row(
    row: Mapping[str, Any], raw_relative: str, raw_sha: str, label: str
) -> dict[str, Any]:
    compact = {
        key: value for key, value in row.items() if key not in REMOVED_FIELDS_SET
    }
    for field, expected in (("raw_shard_path", raw_relative), ("raw_shard_sha256", raw_sha)):
        if field in compact and not _strict_equal(compact[field], expected):
            raise PublicationError(f"{label}: existing {field} disagrees with derived provenance")
        compact[field] = expected

    if "recovery_action_hashes" in row:
        hashes = row["recovery_action_hashes"]
        if not isinstance(hashes, list):
            raise PublicationError(f"{label}: recovery_action_hashes must be a list")
        derived = hashlib.sha256(_canonical_json(hashes)).hexdigest()
        for field, expected in (
            ("action_stream_hash", derived),
            ("action_stream_hash_derived", True),
            ("action_stream_hash_derivation", ACTION_STREAM_DERIVATION),
        ):
            if field in compact and not _strict_equal(compact[field], expected):
                raise PublicationError(f"{label}: existing {field} disagrees with derived action hash")
            compact[field] = expected
    return compact


def _temporary_path(path: Path, ordinal: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(f".{path.name}.publish-{os.getpid()}-{ordinal}.tmp")


def _remove(paths: Iterable[Path]) -> None:
    for path in paths:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            # A failed cleanup must not make us overwrite source or existing
            # publication files; the caller still reports BLOCKED.
            pass


def _process_file(
    source_root: Path,
    output_phase: Path,
    phase: str,
    filename: str,
    ordinal: int,
    seen: set[tuple[Any, ...]],
) -> tuple[dict[str, Any], list[Path]]:
    source_path = source_root / phase / filename
    if not source_path.is_file():
        raise PublicationError(f"missing source row file: {source_path}")
    compact_path = output_phase / filename
    full_path = output_phase / (filename[: -len(".jsonl")] + ".full.jsonl.gz")
    compact_tmp = _temporary_path(compact_path, ordinal)
    full_tmp = _temporary_path(full_path, ordinal)
    temporary = [compact_tmp, full_tmp]
    source_hash = hashlib.sha256()
    compact_hash = hashlib.sha256()
    rows = 0
    try:
        with source_path.open("rb") as source, compact_tmp.open("wb") as compact_stream:
            with full_tmp.open("wb") as full_stream:
                with gzip.GzipFile(
                    filename="", mode="wb", fileobj=full_stream,
                    compresslevel=9, mtime=0,
                ) as full_gzip:
                    for line_number, raw_line in enumerate(source, 1):
                        source_hash.update(raw_line)
                        full_gzip.write(raw_line)
                        if not raw_line.strip():
                            raise PublicationError(f"{source_path}:{line_number}: blank JSONL row")
                        try:
                            value = json.loads(raw_line.decode("utf-8"))
                        except (UnicodeError, TypeError, ValueError, json.JSONDecodeError) as exc:
                            raise PublicationError(
                                f"{source_path}:{line_number}: invalid JSONL row: {exc}"
                            ) from exc
                        if not isinstance(value, Mapping):
                            raise PublicationError(f"{source_path}:{line_number}: row is not an object")
                        row = dict(value)
                        label = f"{source_path}:{line_number}"
                        key = _identity_key(row, label)
                        if key in seen:
                            raise PublicationError(f"{label}: duplicate branch identity")
                        seen.add(key)
                        _validate_status(row, label)
                        if phase == "phase1" and row.get("split") != "calibration":
                            raise PublicationError(f"{label}: Phase-1 row split is not calibration")
                        if phase == "phase2" and row.get("split") not in ("calibration", "evaluation"):
                            raise PublicationError(f"{label}: invalid Phase-2 row split")
                        raw_path, raw_relative = _raw_shard_path(source_root, phase, row, label)
                        raw_sha = _read_raw_shard(raw_path, row, label)
                        projected = _compact_row(row, raw_relative, raw_sha, label)
                        encoded = _json_line(projected)
                        compact_stream.write(encoded)
                        compact_hash.update(encoded)
                        rows += 1
                # The gzip footer is written when its context closes.  Flush
                # and fsync only after that close so the immutable file is
                # durable and hash-verifiable before publication.
                full_stream.flush()
                os.fsync(full_stream.fileno())
            compact_stream.flush()
            os.fsync(compact_stream.fileno())
        if rows <= 0:
            raise PublicationError(f"{source_path}: row file is empty")

        source_sha = source_hash.hexdigest()
        compact_sha = compact_hash.hexdigest()
        gzip_sha, gzip_bytes = _sha256_stream(full_tmp)
        decompressed_sha, decompressed_bytes = _sha256_gzip_uncompressed(full_tmp)
        source_bytes = source_path.stat().st_size
        if decompressed_sha != source_sha or decompressed_bytes != source_bytes:
            raise PublicationError(f"{full_path}: decompressed bytes do not equal original source")
        compact_bytes = compact_tmp.stat().st_size
        if compact_bytes >= MAX_OUTPUT_BYTES or gzip_bytes >= MAX_OUTPUT_BYTES:
            raise PublicationError(
                f"{filename}: projected output exceeds {MAX_OUTPUT_BYTES} bytes "
                f"(compact={compact_bytes}, full_gzip={gzip_bytes})"
            )
        info = {
            "source_file": (Path(phase) / filename).as_posix(),
            "compact_file": (Path(phase) / filename).as_posix(),
            "full_gzip_file": (Path(phase) / full_path.name).as_posix(),
            "row_count": rows,
            "full_sha256": source_sha,
            "compact_sha256": compact_sha,
            "gzip_sha256": gzip_sha,
            "full_gzip_sha256": gzip_sha,
            "source_bytes": source_bytes,
            "compact_bytes": compact_bytes,
            "full_gzip_bytes": gzip_bytes,
            "decompressed_full_bytes": decompressed_bytes,
            "decompressed_full_sha256": decompressed_sha,
        }
        return info, temporary
    except Exception:
        _remove(temporary)
        raise

def _existing_matches(path: Path, expected_sha: str, expected_size: int) -> bool:
    if not path.exists():
        return False
    if not path.is_file():
        raise PublicationError(f"publication destination is not a file: {path}")
    actual_sha, actual_size = _sha256_stream(path)
    if actual_sha != expected_sha or actual_size != expected_size:
        raise PublicationError(f"immutable publication differs: {path}")
    return True


def _install_immutable(temp: Path, destination: Path, expected_sha: str, expected_size: int) -> None:
    if _existing_matches(destination, expected_sha, expected_size):
        _remove((temp,))
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temp, destination)


def _admit_phase2(source_root: Path) -> dict[str, Any]:
    """Call the canonical diagnostic admission API before opening any row."""
    if matrix is None:
        raise PublicationError("evaluation OPEN_DENY: matrix module unavailable")
    admit = getattr(matrix, "selected_diagnostic_budget", None)
    if not callable(admit):
        raise PublicationError("evaluation OPEN_DENY: selected_diagnostic_budget unavailable")
    try:
        result = admit(source_root)
    except Exception as exc:
        raise PublicationError(f"evaluation OPEN_DENY: {exc}") from exc
    if not isinstance(result, Mapping):
        raise PublicationError("evaluation OPEN_DENY: admission returned no mapping")
    selected = result.get("selected_budget", result)
    if not isinstance(selected, Mapping):
        raise PublicationError("evaluation OPEN_DENY: admission has no selected budget")
    # Keep only JSON-safe, useful admission evidence.  The matrix call itself
    # remains the source of authorization; this receipt is not a second gate.
    budget = {}
    for name in ("tail_horizon", "action_budget", "policy_call_cap"):
        if name in selected:
            budget[name] = selected[name]
    if len(budget) != 3:
        raise PublicationError("evaluation OPEN_DENY: admission budget is incomplete")
    evidence: dict[str, Any] = {"authorized": True, "selected_budget": budget}
    for name in (
        "selection_receipt_sha256",
        "diagnostic_selection_receipt_sha256",
        "protocol_sha256",
    ):
        value = result.get(name)
        if isinstance(value, str):
            evidence[name] = value
    return evidence


def _receipt_bytes(
    phase: str,
    source_root: Path,
    output_root: Path,
    files: Sequence[Mapping[str, Any]],
    admission: Mapping[str, Any] | None,
) -> bytes:
    value = {
        "schema_version": 1,
        "phase": phase,
        "status": "COMPLETE",
        "blocked": False,
        "source_root": str(source_root),
        "repo_output_root": str(output_root),
        "files": [dict(item) for item in files],
        "projection": {
            "removed_fields": list(REMOVED_FIELDS),
            "action_stream_hash_derivation": ACTION_STREAM_DERIVATION,
            "raw_shards_preserved": True,
            "full_jsonl_gzip_is_lossless": True,
        },
        "raw_shards": {
            "root": (Path(phase) / "shards").as_posix(),
            "source_rows_retained": True,
        },
        "evaluation_read": phase == "phase2",
        "diagnostic_authorization": dict(admission) if admission is not None else None,
    }
    return _canonical_json(value) + b"\n"


def publish_rows(
    source_root: str | Path,
    repo_output_root: str | Path,
    phase: str,
) -> dict[str, Any]:
    """Project all standard row files for one phase into a publication root.

    The return value follows the repository's fail-closed convention: malformed
    or incomplete input returns ``status=BLOCKED`` and leaves no new output row
    files.  Existing immutable outputs are accepted only when their bytes and
    receipt metadata match exactly.
    """
    if phase not in ROW_FILES:
        return {"schema_version": 1, "phase": phase, "status": "BLOCKED", "blocked": True,
                "blocking_reasons": [f"unsupported phase: {phase!r}"]}
    source_path = Path(source_root)
    output_path = Path(repo_output_root)
    admission: dict[str, Any] | None = None
    temporary: list[Path] = []
    try:
        # This call intentionally precedes source-root checks and every row
        # file read.  It is the only Phase-2 opening gate.
        if phase == "phase2":
            admission = _admit_phase2(source_path)
        source_resolved = source_path.resolve()
        output_resolved = output_path.resolve()
        if source_resolved == output_resolved:
            raise PublicationError("source-root and repo-output-root must be different")
        if not source_resolved.is_dir():
            raise PublicationError(f"source root does not exist: {source_resolved}")
        output_phase = output_resolved / phase
        seen: set[tuple[Any, ...]] = set()
        infos: list[dict[str, Any]] = []
        # Generate every output into temporary files first.  A failure in the
        # second standard row file therefore cannot publish a partial phase.
        for ordinal, filename in enumerate(ROW_FILES[phase]):
            info, paths = _process_file(
                source_resolved, output_phase, phase, filename, ordinal, seen
            )
            infos.append(info)
            temporary.extend(paths)

        receipt_path = output_phase / RECEIPT_NAME
        receipt_tmp = _temporary_path(receipt_path, len(ROW_FILES[phase]) + 1)
        temporary.append(receipt_tmp)
        receipt_raw = _receipt_bytes(phase, source_resolved, output_resolved, infos, admission)
        receipt_tmp.write_bytes(receipt_raw)
        receipt_sha = hashlib.sha256(receipt_raw).hexdigest()
        receipt_size = len(receipt_raw)
        if receipt_size >= MAX_OUTPUT_BYTES:
            raise PublicationError("publication receipt exceeds output size limit")

        # Check all existing destinations before installing any missing one.
        destinations: list[tuple[Path, Path, str, int]] = []
        for ordinal, info in enumerate(infos):
            compact_dest = output_resolved / info["compact_file"]
            full_dest = output_resolved / info["full_gzip_file"]
            # Use the same deterministic temporary names as the process step
            # rather than relying on source-root paths in the receipt.
            compact_temp = _temporary_path(compact_dest, ordinal)
            full_temp = _temporary_path(full_dest, ordinal)
            destinations.extend(
                (
                    (compact_dest, compact_temp, info["compact_sha256"], int(info["compact_bytes"])),
                    (full_dest, full_temp, info["gzip_sha256"], int(info["full_gzip_bytes"])),
                )
            )
        # _process_file's temporary names use the same destination/ordinal.
        destinations.append((receipt_path, receipt_tmp, receipt_sha, receipt_size))
        for destination, temp, expected_sha, expected_size in destinations:
            if not _existing_matches(destination, expected_sha, expected_size):
                # Destination is absent; installation below will use the temp
                # path.  The check is kept here to fail before any install.
                pass
        for destination, temp, expected_sha, expected_size in destinations:
            _install_immutable(temp, destination, expected_sha, expected_size)
            if temp in temporary:
                temporary.remove(temp)

        return {
            "schema_version": 1,
            "phase": phase,
            "status": "COMPLETE",
            "blocked": False,
            "source_root": str(source_resolved),
            "repo_output_root": str(output_resolved),
            "files": infos,
            "receipt": str(receipt_path),
            "receipt_sha256": receipt_sha,
            "row_count": sum(int(info["row_count"]) for info in infos),
            "evaluation_read": phase == "phase2",
            "diagnostic_authorization": admission,
        }
    except Exception as exc:
        _remove(temporary)
        return {
            "schema_version": 1,
            "phase": phase,
            "status": "BLOCKED",
            "blocked": True,
            "evaluation_read": False if phase == "phase2" and admission is None else phase == "phase2",
            "blocking_reasons": [str(exc)],
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--repo-output-root", required=True, type=Path)
    parser.add_argument("--phase", choices=tuple(ROW_FILES), required=True)
    args = parser.parse_args(argv)
    result = publish_rows(args.source_root, args.repo_output_root, args.phase)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result.get("status") == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
