# R16-P14 LIBERO Stage-1b

Stage-1b repairs the metric and replay contracts before deciding whether the
earliest-irreversible-prefix mechanism deserves any learned risk model. It is
strictly an oracle/mechanism audit: no risk head, VLA, world model, or final
algorithm claim is permitted.

The frozen parent is repository commit
`ee56aa096e308214c38132d0e6d2a9e576c29792` (tree
`50e56b84c5e867447b22f81e31454646a12a9eb8`). Stage-1 evidence under
`artifacts/formal_pilot` and `artifacts/checkpoints` is read-only input.

Run the standard-library-only offline reanalysis with:

```bash
experiments/r16_p14_libero_stage1b/commands.sh phase-a
```

The reconstruction replay gate requires the existing LIBERO simulation
environment and frozen Bowl checkpoint:

```bash
R16P14_PYTHON=/path/to/libero/python \
  experiments/r16_p14_libero_stage1b/commands.sh phase-b
```

Every later phase is conditional on the preceding preregistered gate.

## Final result

The final decision is **`KILL_CORE_HYPOTHESIS`**. Phase A removes the old Bowl
positive under the corrected safety contract. Phase B repairs Bowl replay to
100% across 180 branch points. Phase C then evaluates 44 fixed calibration
configurations without BC: only the single bounded replacement task qualifies,
while the gate requires at least two tasks. Evaluation demos remain untouched,
and Phases D/E were not run.

Run Phase C and its deterministic aggregation with:

```bash
R16P14_PYTHON=/path/to/libero/python \
  experiments/r16_p14_libero_stage1b/commands.sh phase-c-calibrate
R16P14_PYTHON=/path/to/libero/python \
  experiments/r16_p14_libero_stage1b/commands.sh phase-c-summarize
```

See [`reports/REPORT.md`](reports/REPORT.md) for the evidence-layered result.
