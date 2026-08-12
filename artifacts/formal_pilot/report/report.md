# R16-P14 Stage-1 Oracle Prefix Feasibility Audit — LIBERO pilot

Decision: `REVISE_TASKS_OR_PERTURBATIONS`

## Clean baseline

| Task | h=1 | h=4 | h=8 | h=16 |
| --- | ---: | ---: | ---: | ---: |
| Open middle drawer / mechanism obstruction | 3.3% | 10.0% | 10.0% | 10.0% |
| Bowl on plate / target shift | 6.7% | 6.7% | 6.7% | 0.0% |
| Wine bottle on rack / grasp slip | 3.3% | 16.7% | 16.7% | 0.0% |

## Oracle prefix audit

| Task | Candidates | Usable unsafe | Median window | Safe-prefix retention | Replay |
| --- | ---: | ---: | ---: | ---: | ---: |
| Open middle drawer / mechanism obstruction | 30 | 27 | 0 | 0.0% | 100.0% |
| Bowl on plate / target shift | 30 | 17 | 4 | 50.0% | 86.7% |
| Wine bottle on rack / grasp slip | 30 | 28 | 0.0 | 0.0% | 100.0% |

## Gates

- FAIL — `minimum_30_usable_per_task`
- FAIL — `at_least_two_tasks_median_window_ge_2`
- FAIL — `relative_unsafe_outcome_reduction_ge_30pct`
- FAIL — `relative_rework_reduction_ge_20pct`
- FAIL — `median_safe_prefix_retention_ge_30pct`
- PASS — `two_distinct_cause_operator_winners`
- FAIL — `replay_determinism_ge_99pct`

## Limitations

- The clean pilot uses 10 rather than the preregistered final 50 episodes per seed and horizon.
- The oracle audit uses policy seed 7; the clean baseline covers all three policy seeds.
- Physical recoverability is a privileged scripted-recovery proxy, not an exhaustive dynamics proof.
- The policy is state-observation chunked BC; no learned risk head, VLA, or world model is evaluated.
