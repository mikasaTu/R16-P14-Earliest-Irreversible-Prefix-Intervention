# Phase A — corrected offline reanalysis

No simulator or GPU was used. All 90 immutable Stage-1 oracle records were reread.

## Old-to-new comparison

| Task | Old mixed unsafe | Explicit cause violations | Failure-only inclusions | Old median window | Corrected median window | Old retention | Corrected post-detection retention | Paired n | Signal survives |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Drawer | 27 | 6 | 21 | 0 | 0.0 | 0.0% | 0.0% | 0 | NO |
| Bowl | 17 | 5 | 12 | 4 | 0 | 50.0% | 0.0% | 2 | NO |
| Wine | 28 | 28 | 0 | 0.0 | 0.0 | 0.0% | 0.0% | 4 | NO |

## Bowl conclusion

The Stage-1 Bowl safety signal does **not** survive the corrected primary definition. Only 5 of 30 candidates had an explicit target-cause violation at no intervention, versus 17 old mixed-unsafe candidates. The corrected median window is 0 and median post-detection retention is 0.0%. Only 2 complete M0/M1/M2 pairs exist. On those pairs, M0/M1/M2 cause-violation counts are 0/1/1; immediate full replanning was not beaten on safety.

M1 does reduce newly executed non-nominal actions on this tiny paired subset, but that observation is not a positive safety result and is too underpowered to carry the hypothesis.

## Evidence limitations

- Stage-1 did not persist nominal action values, so corrected total path length cannot be reconstructed offline.
- Paired metrics are defined only when an explicit-cause candidate has a deployable successful intervention.
- These are deterministic re-labelings of existing branches, not newly executed counterfactuals.
- Phase B must independently repair and test replay before any expert calibration.
