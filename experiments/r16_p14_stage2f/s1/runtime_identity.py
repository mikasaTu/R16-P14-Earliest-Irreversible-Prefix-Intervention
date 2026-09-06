"""Verify actual imported modules against the pre-data compatible runtime receipt."""
from __future__ import annotations
import importlib
import sys
from pathlib import Path
from .common import ROOT, file_sha, read_json_retry

def verify_runtime():
    receipt_path = ROOT / "artifacts/stage2f/preflight/runtime_compat/runtime_compat_receipt.json"
    receipt = read_json_retry(receipt_path)
    if sys.prefix != receipt["runtime"]["prefix"]:
        raise RuntimeError("S1 compatible runtime prefix drift")
    observed = {}
    for name, expected in receipt["package_versions_and_paths"].items():
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        expected_path = (ROOT / name.replace(".", "/") / "__init__.py"
                         if name.startswith("libero") else Path(expected["file"]))
        version = getattr(module, "__version__", None)
        actual = {"file": str(path), "version": version, "sha256": file_sha(path)}
        observed[name] = actual
        if path != expected_path.resolve() or version != expected["version"] or actual["sha256"] != expected["file_sha256"]:
            raise RuntimeError(f"S1 actual imported module drift: {name}")
    torch = importlib.import_module("torch")
    if torch.version.cuda != "12.4" or torch.__version__ != "2.6.0+cu124":
        raise RuntimeError("S1 torch/CUDA build mismatch")
    for name in ("pth", "torch_version", "env_manifest"):
        pin = receipt["content_pins"][name]
        if file_sha(pin["path"]) != pin["sha256"]:
            raise RuntimeError(f"S1 runtime content drift: {name}")
    for name, binding in receipt["bindings"].items():
        if binding.get("is_symlink"):
            path = Path(binding["path"])
            if not path.is_symlink() or str(path.resolve()) != binding["resolved_target"]:
                raise RuntimeError(f"S1 runtime binding drift: {name}")
    return {"receipt_sha256": file_sha(receipt_path), "prefix": sys.prefix,
            "python_version": sys.version, "cuda_build": torch.version.cuda,
            "actual_imports": observed}
