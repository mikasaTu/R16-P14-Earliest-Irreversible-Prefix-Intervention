# Safe-Replanability Atlas preflight

Status: **PREFLIGHT_ONLY_NOT_LAUNCHED**.

This schedule is derived only from frozen calibration qualification and the clean actor gate. It does not inspect evaluation or held-out outcomes and does not execute a branch.

Eligible task intersection: `2` tasks.
Frozen events: `100`.
Primary R(k) branches: `4200`.
Maximum secondary operator branches: `2700`.
Maximum total branch runs: `6900` (source-prefix reconstruction overhead excluded).

| Task | Primary severity | Secondary severity | Evaluation | Held-out |
| --- | --- | --- | ---: | ---: |
| put_the_bowl_on_the_stove | lead14_lateral050mm | lead14_lateral075mm | 30 | 20 |
| put_the_cream_cheese_in_the_bowl | lead08_shift040mm | lead10_shift060mm | 30 | 20 |

The seven primary method readouts are selected from the same exhaustive R(k) branches and therefore do not multiply the primary branch count. Secondary controls are separately budgeted at d, k_best, and k_last_safe only.
