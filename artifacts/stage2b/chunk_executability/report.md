# Phase A — action-chunk executability

Status: **PASS**; `H_valid=16` (required >=8).

| horizon | success | late reach | prefix faithful | errors | passes |
|---:|---:|---:|---:|---:|:---:|
| 1 | 0.667 | 0.708 | 0.988 | 0 | yes |
| 2 | 0.850 | 0.925 | 0.991 | 0 | yes |
| 4 | 0.842 | 0.892 | 1.000 | 0 | yes |
| 8 | 0.883 | 0.933 | 1.000 | 0 | yes |
| 16 | 0.933 | 0.950 | 1.000 | 0 | yes |

A blocked result identifies an unsuitable frozen chunk substrate; it is not interpreted as an R16-P14 mechanism failure.
Object-drop metrics use target-aware monitor v2; the height-only first pass was invalidated before Phase B (see the monitor-bug audit).
The explicit user override requires later phases to run regardless of this gate.
