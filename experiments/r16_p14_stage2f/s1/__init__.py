"""Stage-2F S1 zero-injection recovery backend."""

from .dispatch import dispatch_branch, execute_spawned_branch, run_spawned_branch
from .measurement import (
    SimulationStepRecorder,
    label_trace,
    record_step,
    trace_content_sha256,
    write_trace_gzip_atomic,
)
from .runtime import (
    OPERATORS,
    REFERENCE_OPERATORS,
    RuntimeBlocked,
    execute_branch,
    execute_reference_branch,
    execution_contract,
)

__all__ = [
    "OPERATORS", "REFERENCE_OPERATORS", "RuntimeBlocked", "execution_contract",
    "execute_branch", "execute_reference_branch", "run_spawned_branch",
    "dispatch_branch", "execute_spawned_branch", "SimulationStepRecorder",
    "record_step", "label_trace", "trace_content_sha256", "write_trace_gzip_atomic",
]
