from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pai import project_published_rows as publisher  # noqa: E402

TASK_CREAM = "put_the_cream_cheese_in_the_bowl"
TASK_BOWL = "put_the_bowl_on_the_plate"


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _row(
    task: str = TASK_CREAM,
    *,
    event: str = "event-1",
    init: str = "11",
    split: str = "calibration",
    operator: str = "fresh_h4",
    recovery_seed: int = 7,
    trace_name: str = "branch-1.jsonl.gz",
    status: str = "COMPLETE",
) -> dict[str, object]:
    return {
        "event_instance_id": event,
        "task": task,
        "init_state_id": init,
        "split": split,
        "generator_actor_seed": 7,
        "recovery_actor_seed": recovery_seed,
        "operator": operator,
        "prefix_k": 2,
        "tail_horizon": 4,
        "action_budget": 8,
        "policy_call_cap": 8,
        "safe_success": 1.0,
        "pid": 123,
        "env_hash": "env-hash",
        "chunk_hash": "chunk-hash",
        "status": status,
        "trace_path": f"/external/{split}/contact_topology/{task}/{trace_name}",
        "recovery_action_hashes": ["a" * 64, "b" * 64],
        "recovery_chunk_hashes": ["c" * 64],
        "d4_signatures": {"all": "large-d4"},
        "labels": {"label_complete": True, "record_count": 1},
        "topology_candidate": {"pairs": [["a", "b"]]},
        "nested": {"preserve": [1, "x", None]},
    }


def _make_source(root: Path, phase: str = "phase1") -> tuple[Path, list[dict[str, object]]]:
    names = ("grid_rows.jsonl", "reference_rows.jsonl") if phase == "phase1" else ("atlas_rows.jsonl", "reference_rows.jsonl")
    rows: list[dict[str, object]] = [
        _row(),
        _row(
            task=TASK_BOWL,
            event="event-2",
            init="12",
            operator="immediate_fresh",
            trace_name="branch-2.jsonl.gz",
        ),
    ]
    # Keep branch identities unique across the two standard files.
    ref = _row(
        task=TASK_CREAM,
        event="event-ref",
        init="13",
        operator="fixed_delay_2",
        trace_name="branch-ref.jsonl.gz",
    )
    ref["is_reference"] = True
    rows_by_file = ([rows[0]], [ref]) if phase == "phase1" else ([rows[0]], [ref])
    for filename, material in zip(names, rows_by_file):
        source = root / phase / filename
        source.parent.mkdir(parents=True, exist_ok=True)
        # Deliberately use noncanonical spacing so the lossless full member is
        # observably the original source bytes, while compact JSON is stable.
        raw = b"".join(
            (json.dumps(item, ensure_ascii=False) + " \n").encode("utf-8")
            for item in material
        )
        source.write_bytes(raw)
        for item in material:
            trace_name = Path(str(item["trace_path"])).name
            raw_name = trace_name[: -len(".jsonl.gz")] + ".json"
            raw_path = root / phase / "shards" / str(item["task"]) / raw_name
            _write(raw_path, item)
    return root, rows


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_phase1_projection_is_lossless_deterministic_and_provenanced(tmp_path: Path):
    source, _ = _make_source(tmp_path / "source")
    output = tmp_path / "repo"
    result = publisher.publish_rows(source, output, "phase1")
    assert result["status"] == "COMPLETE", result
    phase = output / "phase1"
    receipt = json.loads((phase / publisher.RECEIPT_NAME).read_text(encoding="utf-8"))
    assert receipt["projection"]["full_jsonl_gzip_is_lossless"] is True
    assert receipt["projection"]["removed_fields"] == list(publisher.REMOVED_FIELDS)
    for item in receipt["files"]:
        source_bytes = (source / item["source_file"]).read_bytes()
        full_path = output / item["full_gzip_file"]
        compact_path = output / item["compact_file"]
        assert hashlib.sha256(source_bytes).hexdigest() == item["full_sha256"]
        assert gzip.open(full_path, "rb").read() == source_bytes
        assert hashlib.sha256(full_path.read_bytes()).hexdigest() == item["gzip_sha256"]
        assert hashlib.sha256(compact_path.read_bytes()).hexdigest() == item["compact_sha256"]
        assert item["row_count"] == len(_read_jsonl(compact_path))

    compact = _read_jsonl(phase / "grid_rows.jsonl")[0]
    assert all(name not in compact for name in publisher.REMOVED_FIELDS)
    original = json.loads((source / "phase1" / "grid_rows.jsonl").read_text().splitlines()[0])
    for name, value in original.items():
        if name not in publisher.REMOVED_FIELDS:
            if name not in {"raw_shard_path", "raw_shard_sha256", "action_stream_hash", "action_stream_hash_derived", "action_stream_hash_derivation"}:
                assert compact[name] == value
    expected = hashlib.sha256(
        json.dumps(["a" * 64, "b" * 64], sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    assert compact["action_stream_hash"] == expected
    assert compact["action_stream_hash_derived"] is True
    assert compact["action_stream_hash_derivation"] == publisher.ACTION_STREAM_DERIVATION
    assert compact["raw_shard_path"] == "phase1/shards/put_the_cream_cheese_in_the_bowl/branch-1.json"
    raw_sha = hashlib.sha256(
        (source / compact["raw_shard_path"]).read_bytes()
    ).hexdigest()
    assert compact["raw_shard_sha256"] == raw_sha
    assert compact["safe_success"] == 1.0
    assert compact["nested"] == {"preserve": [1, "x", None]}

    # A second publication from identical source bytes is byte-identical,
    # including the deterministic gzip header and receipt.
    output2 = tmp_path / "repo2"
    result2 = publisher.publish_rows(source, output2, "phase1")
    assert result2["status"] == "COMPLETE", result2
    for name in ("grid_rows.jsonl", "grid_rows.full.jsonl.gz", "reference_rows.jsonl", "reference_rows.full.jsonl.gz"):
        assert (output / "phase1" / name).read_bytes() == (output2 / "phase1" / name).read_bytes()


def test_phase1_rejects_raw_shard_identity_mismatch_without_publishing(tmp_path: Path):
    source, _ = _make_source(tmp_path / "source")
    raw = source / "phase1" / "shards" / TASK_CREAM / "branch-1.json"
    value = json.loads(raw.read_text(encoding="utf-8"))
    value["event_instance_id"] = "wrong-event"
    _write(raw, value)
    result = publisher.publish_rows(source, tmp_path / "repo", "phase1")
    assert result["status"] == "BLOCKED"
    assert "disagrees on event_instance_id" in result["blocking_reasons"][0]
    assert not (tmp_path / "repo" / "phase1" / "grid_rows.jsonl").exists()
    assert not list((tmp_path / "repo" / "phase1").glob("*.tmp")) if (tmp_path / "repo" / "phase1").exists() else True


def test_phase2_calls_diagnostic_admission_before_reading_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source = tmp_path / "source"
    malformed = source / "phase2" / "atlas_rows.jsonl"
    malformed.parent.mkdir(parents=True)
    malformed.write_text("{ this must not be parsed\n", encoding="utf-8")
    calls: list[Path] = []

    def deny(root: Path):
        calls.append(root)
        raise RuntimeError("diagnostic proof absent")

    monkeypatch.setattr(publisher.matrix, "selected_diagnostic_budget", deny, raising=False)
    result = publisher.publish_rows(source, tmp_path / "repo", "phase2")
    assert result["status"] == "BLOCKED"
    assert result["evaluation_read"] is False
    assert calls == [source]
    assert any("OPEN_DENY" in reason for reason in result["blocking_reasons"])
    assert not any("invalid JSONL" in reason for reason in result["blocking_reasons"])


def test_phase2_authorized_projection_uses_phase2_raw_for_reused_trace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source, rows = _make_source(tmp_path / "source", "phase2")
    # The trace path is intentionally phase1-looking: phase2 publication must
    # derive phase2/shards/<task>/<basename>.json from the current phase.
    rows[0]["trace_path"] = "/old/phase1/contact_topology/put_the_cream_cheese_in_the_bowl/branch-1.jsonl.gz"
    source_row = source / "phase2" / "atlas_rows.jsonl"
    source_row.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    raw = source / "phase2" / "shards" / TASK_CREAM / "branch-1.json"
    _write(raw, rows[0])
    monkeypatch.setattr(
        publisher.matrix,
        "selected_diagnostic_budget",
        lambda _root: {"selected_budget": {"tail_horizon": 4, "action_budget": 8, "policy_call_cap": 8}, "selection_receipt_sha256": "a" * 64},
        raising=False,
    )
    result = publisher.publish_rows(source, tmp_path / "repo", "phase2")
    assert result["status"] == "COMPLETE", result
    compact = _read_jsonl(tmp_path / "repo" / "phase2" / "atlas_rows.jsonl")[0]
    assert compact["raw_shard_path"].startswith("phase2/shards/")
    receipt = json.loads((tmp_path / "repo" / "phase2" / publisher.RECEIPT_NAME).read_text())
    assert receipt["diagnostic_authorization"]["authorized"] is True


def test_unknown_error_and_duplicate_identity_fail_closed(tmp_path: Path):
    source, _ = _make_source(tmp_path / "source")
    path = source / "phase1" / "grid_rows.jsonl"
    row = json.loads(path.read_text())
    row["status"] = "BLOCKED"
    row["error_type"] = "Timeout"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    result = publisher.publish_rows(source, tmp_path / "repo", "phase1")
    assert result["status"] == "BLOCKED"
    assert "unknown/error" in result["blocking_reasons"][0]

    source2, _ = _make_source(tmp_path / "source2")
    grid = source2 / "phase1" / "grid_rows.jsonl"
    first = json.loads(grid.read_text())
    grid.write_text(json.dumps(first) + "\n" + json.dumps(first) + "\n", encoding="utf-8")
    result2 = publisher.publish_rows(source2, tmp_path / "repo2", "phase1")
    assert result2["status"] == "BLOCKED"
    assert "duplicate branch identity" in result2["blocking_reasons"][0]


def test_output_size_limit_is_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source, _ = _make_source(tmp_path / "source")
    monkeypatch.setattr(publisher, "MAX_OUTPUT_BYTES", 1)
    result = publisher.publish_rows(source, tmp_path / "repo", "phase1")
    assert result["status"] == "BLOCKED"
    assert "exceeds" in result["blocking_reasons"][0]
    assert not (tmp_path / "repo" / "phase1" / "grid_rows.jsonl").exists()


def test_cli_requires_explicit_paths_and_returns_machine_readable_result(tmp_path: Path):
    source, _ = _make_source(tmp_path / "source")
    code = publisher.main(
        ["--source-root", str(source), "--repo-output-root", str(tmp_path / "repo"), "--phase", "phase1"]
    )
    assert code == 0
