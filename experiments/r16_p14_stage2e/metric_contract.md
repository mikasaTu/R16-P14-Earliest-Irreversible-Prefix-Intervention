# Stage-2E S0 metric contract

S0 is CPU-only, zero-rollout, diagnostic-only offline reanalysis. Independent unit is an event cluster `(task, init_state_id)`; prefix rows are never independent observations. Stage-2C numbers are always reported as `all_contaminated` (96 events) and `replay_valid_subset` (63 events). Bootstrap is 10,000 resamples, seed 216214, percentile 95% CI and cluster-first weighting.

A1 reproduces the frozen evaluation machine table exactly. A2/A3/A4 use calibration only, boundary `None` falls back to `d=2` with `fresh_h16`, and fixed winner candidates are only fixed delays 1/2/4/8 with the preregistered lexicographic tie-break. A5 aggregates raw `prefix_cause_violation` for `any` and `fraction`; only `cause_violation_type` supplies the type set.

B1 is within-stage descriptive alignment only. Stage-2C runtime labels are `shared_runtime_known_invalid`/`shared_runtime_no_detected_violation`; Stage-2D calibration is `fresh_process_verified`. B2 requires at least two configured preset budget levels; event residual horizon or actual usage is not a budget grid. Evaluation files are open-deny until the B2 selection receipt is written, and afterwards are descriptive only.

C recomputes U1/U2/U3/U4 from the complete 4-operator × 3-actor × k=2..16 raw grid. `None` uses sentinel `d-1=1` only for monotonicity. U4 is cross-checked against the existing atlas. D only writes an S1 contract and does not execute it.

All outputs carry `diagnostic_only=true`, `formal_positive_evidence_allowed=false`, `new_idea_generated=false`; PAI planned/submitted jobs are `0/0`.
