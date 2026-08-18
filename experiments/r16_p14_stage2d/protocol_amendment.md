# Stage-2D protocol amendment: diagnostic continuation

This amendment records the outer user instruction that all planned Stage-2D
experiments must be executed even when an upstream scientific gate fails. It
does not alter the preregistered gate, its thresholds, the immutable Stage-1b
or Stage-2C conclusions, or the frozen evaluation rule.

When Phase A--D qualification or event-construction gates fail, every
downstream atlas, rule, confirmatory matrix, oracle appendix, and statistic is
marked `diagnostic_only: true`. It is retained to audit implementation,
controls, resource use, and possible mechanisms, but it is not formal evidence
for a positive hypothesis. In particular,
`formal_positive_evidence_allowed: false` globally for this Stage-2D run.

The continuation must not read evaluation outcomes to tune perturbations,
rules, baselines, or thresholds. A failed upstream gate remains failed, and no
diagnostic result may set `accepted=true`, increase novelty above
`N2_ORACLE_PROTOCOL_BOUNDARY_ONLY`, or use forbidden claims such as physical
irreversibility, VLA validation, a learned intervention policy, N3, or N4.

The amendment is an execution-scope change only. Calibration and evaluation
rows remain separated, event-level clustering is preserved for statistics, and
partial PAI artifacts are not eligible for consolidation or reporting as a
complete matrix.
