"""Verify formal or diagnostic selection receipts without a checkout in PAI.

The formal receipt remains the only default selection artifact.  Diagnostic
continuation uses a separate, fixed repository path and proof so that it
cannot replace the formal gate by renaming or relocating a file.
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

FORMAL_PATH = "artifacts/stage2f/phase1/selection_receipt.json"
DIAGNOSTIC_PATH = "artifacts/stage2f/phase1/diagnostic_selection_receipt.json"
FORMAL_AUTH_PATH = "artifacts/stage2f/phase1/selection_authorization.json"
# Backwards-compatible name used by the formal proof helper.
PATH = FORMAL_PATH
DIAGNOSTIC_AUTH_PATH = "artifacts/stage2f/phase1/diagnostic_selection_authorization.json"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

# Keep this local to selection admission.  matrix.py performs the same
# preregistered range check after receipt verification.
_ALLOWED_TAILS = (4, 8, 16)
_ALLOWED_ACTIONS = (8, 16, 32)


def oid(kind: str, data: bytes) -> str:
    """Return the SHA-1 Git object id for raw object content."""
    return hashlib.sha1(
        kind.encode() + b" " + str(len(data)).encode() + b"\0" + data
    ).hexdigest()


def receipt_path(diagnostic: bool = False) -> str:
    """Return the one Git path admitted for the requested selection mode."""
    return DIAGNOSTIC_PATH if diagnostic else FORMAL_PATH


def authorization_path(diagnostic: bool = False) -> str:
    """Return the matching authorization path for the requested mode."""
    return DIAGNOSTIC_AUTH_PATH if diagnostic else FORMAL_AUTH_PATH


def _check_local_receipt_name(receipt: Path, diagnostic: bool) -> None:
    # Runtime output is rooted outside the Git checkout (root/phase1/...).
    # Bind the local artifact to the fixed basename/phase directory while the
    # Git proof below binds its bytes to the exact repository path.
    expected = Path(receipt_path(diagnostic))
    if receipt.name != expected.name:
        raise RuntimeError("selection receipt path basename is not admitted")
    if diagnostic and receipt.parent.name != expected.parent.name:
        raise RuntimeError("diagnostic receipt is not a phase1 artifact")


def _tree_proof(raw: bytes, authorization: Mapping[str, Any], diagnostic: bool) -> None:
    commit = authorization.get("git_commit", "")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RuntimeError("invalid selection commit")
    try:
        obj = base64.b64decode(
            authorization.get("commit_object_base64", ""), validate=True
        )
    except Exception as exc:
        raise RuntimeError("invalid committed object encoding") from exc
    if oid("commit", obj) != commit:
        raise RuntimeError("selection commit object does not hash to commit")
    root = obj.split(b"\n", 1)[0]
    if not root.startswith(b"tree "):
        raise RuntimeError("commit lacks tree")
    expected = root[5:].decode()
    proof = authorization.get("tree_path_proof", [])
    path = receipt_path(diagnostic)
    parts = path.split("/")
    if not isinstance(proof, list) or len(proof) != len(parts):
        raise RuntimeError("incomplete receipt tree proof")
    for name, part in zip(parts, proof):
        if not isinstance(part, Mapping):
            raise RuntimeError("malformed receipt tree proof entry")
        try:
            data = base64.b64decode(part["object_base64"], validate=True)
        except Exception as exc:
            raise RuntimeError("invalid tree object encoding") from exc
        if oid("tree", data) != expected:
            raise RuntimeError("tree proof hash mismatch")
        cursor = 0
        found = None
        while cursor < len(data):
            try:
                space = data.index(b" ", cursor)
                nul = data.index(b"\0", space)
            except ValueError as exc:
                raise RuntimeError("malformed Git tree object") from exc
            entry_name = data[space + 1 : nul].decode()
            child = data[nul + 1 : nul + 21]
            if len(child) != 20:
                raise RuntimeError("malformed Git tree entry")
            if entry_name == name:
                found = child.hex()
            cursor = nul + 21
        if found is None:
            raise RuntimeError("receipt path absent from committed tree")
        expected = found
    if oid("blob", raw) != expected:
        raise RuntimeError("receipt bytes are not the committed blob")


def verify_commit_proof(
    receipt: str | Path,
    authorization: Mapping[str, Any],
    diagnostic: bool = False,
) -> bool:
    """Verify receipt bytes, Git object proof, and the fixed selection mode.

    The two-argument formal call remains backwards compatible.  Diagnostic
    verification must opt in explicitly and must carry the diagnostic Git
    path in its authorization; an arbitrary receipt path or a formal proof
    cannot be used for the diagnostic atlas.
    """
    receipt = Path(receipt)
    _check_local_receipt_name(receipt, diagnostic)
    raw = receipt.read_bytes()
    if hashlib.sha256(raw).hexdigest() != authorization.get("selection_receipt_sha256"):
        raise RuntimeError("receipt SHA256 mismatch")
    expected_path = receipt_path(diagnostic)
    selection_path = authorization.get("selection_path")
    if diagnostic:
        if selection_path != expected_path:
            raise RuntimeError("diagnostic authorization path mismatch")
        if authorization.get("diagnostic_atlas") is not True:
            raise RuntimeError("diagnostic authorization flag missing")
        if not isinstance(authorization.get("verified_origin_main"), str) or not re.fullmatch(r"[0-9a-f]{40}", authorization["verified_origin_main"]):
            raise RuntimeError("diagnostic origin readback missing")
    elif selection_path not in (None, expected_path):
        raise RuntimeError("formal authorization path mismatch")
    if authorization.get("diagnostic_atlas") is True and not diagnostic:
        raise RuntimeError("diagnostic proof requires explicit diagnostic mode")
    _tree_proof(raw, authorization, diagnostic)
    return True


def validate_diagnostic_receipt(
    receipt: Mapping[str, Any],
    *,
    protocol_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate the continuation receipt without reading evaluation artifacts.

    The diagnostic-selection worker uses all_candidates and
    prerequisite_failures and numbers the original rank from one.  The aliases
    accepted here also make the admission boundary explicit for a future
    calibration-only writer without weakening budget or proof checks.
    """
    required = (
        "status",
        "selection_source",
        "confirmatory",
        "selected_budget",
        "original_rank",
        "input_sha256",
        "protocol_sha256",
    )
    missing = [name for name in required if name not in receipt]
    if missing:
        raise RuntimeError("diagnostic selection fields missing: " + ",".join(missing))
    if receipt.get("status") != "DIAGNOSTIC_SELECTED":
        raise RuntimeError("diagnostic selection status invalid")
    if receipt.get("selection_source") != "calibration_only":
        raise RuntimeError("diagnostic selection source invalid")
    if receipt.get("confirmatory") is not False:
        raise RuntimeError("diagnostic selection must be non-confirmatory")
    prerequisites = receipt.get(
        "missing_prerequisites", receipt.get("prerequisite_failures")
    )
    if not isinstance(prerequisites, list):
        raise RuntimeError("diagnostic prerequisite list missing")
    input_sha = receipt.get("input_sha256")
    protocol = receipt.get("protocol_sha256")
    if not isinstance(input_sha, str) or not _SHA256.fullmatch(input_sha):
        raise RuntimeError("diagnostic input SHA256 missing or malformed")
    if not isinstance(protocol, str) or not _SHA256.fullmatch(protocol):
        raise RuntimeError("diagnostic protocol SHA256 missing or malformed")
    if protocol_sha256 is not None and protocol != protocol_sha256:
        raise RuntimeError("diagnostic protocol SHA256 mismatch")
    budget = receipt.get("selected_budget")
    if not isinstance(budget, Mapping):
        raise RuntimeError("diagnostic selected_budget missing")
    try:
        normalized = {
            "tail_horizon": int(budget["tail_horizon"]),
            "action_budget": int(budget["action_budget"]),
            "policy_call_cap": int(budget["policy_call_cap"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("diagnostic selected_budget malformed") from exc
    if (
        normalized["tail_horizon"] not in _ALLOWED_TAILS
        or normalized["action_budget"] not in _ALLOWED_ACTIONS
        or normalized["policy_call_cap"] != 8
    ):
        raise RuntimeError("diagnostic selected_budget outside preregistered grid")
    raw_rank = receipt["original_rank"]
    try:
        original_rank = int(raw_rank)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("diagnostic original_rank malformed") from exc
    if isinstance(raw_rank, bool) or original_rank < 1:
        raise RuntimeError("diagnostic original_rank malformed")
    ranked = receipt.get("ranked_candidates", receipt.get("all_candidates"))
    if not isinstance(ranked, list) or not ranked:
        raise RuntimeError("diagnostic ranked_candidates missing")
    selected = receipt.get("selected_candidate")
    if not isinstance(selected, Mapping):
        raise RuntimeError("diagnostic selected_candidate missing")
    if selected.get("budget") != normalized:
        raise RuntimeError("diagnostic selected budget/selected candidate mismatch")
    selected_rank = selected.get("original_rank", selected.get("rank"))
    try:
        selected_rank = int(selected_rank)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("diagnostic selected candidate rank missing") from exc
    if selected_rank != original_rank:
        raise RuntimeError("diagnostic selected budget/original rank mismatch")
    matching = []
    for candidate in ranked:
        if not isinstance(candidate, Mapping):
            continue
        try:
            candidate_rank = int(candidate.get("original_rank", candidate.get("rank", -1)))
        except (TypeError, ValueError):
            continue
        if candidate_rank == original_rank:
            matching.append(candidate)
    if len(matching) != 1 or matching[0].get("budget") != normalized:
        raise RuntimeError("diagnostic original rank is not bound to ranked candidates")
    return {
        **dict(receipt),
        "selected_budget": normalized,
        "original_rank": original_rank,
        "missing_prerequisites": list(prerequisites),
        "ranked_candidates": ranked,
    }
