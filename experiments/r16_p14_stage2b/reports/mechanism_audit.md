# Code-first mechanism reverse audit

This audit explains realized increases and decreases from the executed code paths; it does not generate a new idea.

## Action-chunk substrate

### put_the_cream_cheese_in_the_bowl

| h | success Δ vs h1 | drop | wrong release | regression | disagreement p95 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.000 | 0.033 | 0.033 | 0.067 | n/a |
| 2 | 0.267 | 0.000 | 0.017 | 0.033 | 0.179 |
| 4 | 0.267 | 0.017 | 0.017 | 0.067 | 0.228 |
| 8 | 0.317 | 0.017 | 0.017 | 0.000 | 0.231 |
| 16 | 0.433 | 0.000 | 0.017 | 0.000 | 0.233 |

### put_the_bowl_on_the_stove

| h | success Δ vs h1 | drop | wrong release | regression | disagreement p95 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.000 | 0.000 | 0.000 | 0.000 | n/a |
| 2 | 0.100 | 0.017 | 0.017 | 0.017 | 0.135 |
| 4 | 0.083 | 0.000 | 0.000 | 0.000 | 0.140 |
| 8 | 0.117 | 0.000 | 0.000 | 0.000 | 0.143 |
| 16 | 0.100 | 0.000 | 0.000 | 0.000 | 0.161 |

## Within-event Atlas counterfactuals

- `A_original_vs_B0`: {'safe_success_gain': 13, 'equal_outcome_efficiency_gain': 15, 'cause_safety_loss': 3, 'safe_success_loss': 5, 'cause_safety_gain': 4}; mean new-action difference `-57.525`, mean policy-call difference `-57.525`, equal budgets `True`.
- `N_oracle_vs_A_original`: {'no_observed_difference': 33, 'safe_success_gain': 7}; mean new-action difference `-23.250`, mean policy-call difference `-23.250`, equal budgets `True`.

The concrete gain path is cached task-directed suffix execution before h=1 ACT feedback, which can reduce rework and policy calls. The concrete loss path is the same continuation crossing an absorbing misaligned-release/contact boundary before feedback. Oracle gains additionally contain post-hoc branch-selection advantage and therefore are not deployable evidence.

Operator audit: `NO_OPERATOR_ROUTING_SIGNAL`; local repair: `NO_SIGNAL`.
