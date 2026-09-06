"""Stage-2F S1 backend namespace with lazy exports.

The repository's shared entrypoints add Stage-2A--D package roots at runtime;
therefore importing this namespace must not eagerly import runtime/torch or any
stage package before that bootstrap has run.
"""

from __future__ import annotations

import importlib

_EXPORTS = {
    "OPERATORS": (".runtime", "OPERATORS"),
    "REFERENCE_OPERATORS": (".runtime", "REFERENCE_OPERATORS"),
    "RuntimeBlocked": (".runtime", "RuntimeBlocked"),
    "execution_contract": (".runtime", "execution_contract"),
    "execute_branch": (".runtime", "execute_branch"),
    "execute_reference_branch": (".runtime", "execute_reference_branch"),
    "run_spawned_branch": (".dispatch", "run_spawned_branch"),
    "dispatch_branch": (".dispatch", "dispatch_branch"),
    "execute_spawned_branch": (".dispatch", "execute_spawned_branch"),
    "SimulationStepRecorder": (".measurement", "SimulationStepRecorder"),
    "record_step": (".measurement", "record_step"),
    "label_trace": (".measurement", "label_trace"),
    "trace_content_sha256": (".measurement", "trace_content_sha256"),
    "write_trace_gzip_atomic": (".measurement", "write_trace_gzip_atomic"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    value = getattr(importlib.import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
