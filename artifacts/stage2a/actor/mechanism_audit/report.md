# Post-hoc actor mechanism audit

This is a code-level reverse-engineering audit, not a new idea, model, hyperparameter search, actor gate, Track-B diagnostic probe, or deployable method. One input factor is masked at a time in frozen checkpoints. Teacher-forced errors use deterministic training-cache rows, so they establish representation dependence but do not by themselves establish rollout causality.

## Matched clean-rollout comparison

| Task | ACT success | Weak horizon-1 success | Difference | Seed range | Phase bottleneck |
| --- | ---: | ---: | ---: | ---: | --- |
| put_the_cream_cheese_in_the_bowl | 0.433 | n/a | n/a | 0.400 | late_approach_or_contact_entry |
| put_the_bowl_on_the_plate | 0.333 | 0.06666666666666667 | 0.26666666666666666 | 0.100 | lift_or_transport_transition |
| open_the_middle_drawer_of_the_cabinet | 0.233 | 0.03333333333333333 | 0.2 | 0.200 | early_grasp_or_articulation_entry |
| open_the_top_drawer_and_put_the_bowl_inside | 0.133 | n/a | n/a | 0.200 | early_grasp_or_articulation_entry |
| push_the_plate_to_the_front_of_the_stove | 0.400 | n/a | n/a | 0.900 | late_approach_or_contact_entry |
| put_the_bowl_on_the_stove | 0.900 | n/a | n/a | 0.200 | no_single_phase_bottleneck_at_gate_resolution |

## Frozen-input ablations

Positive Δ means the ablation increased deployed first-action position MAE relative to the full actor; negative Δ means it reduced that offline error.

| Task | Repeat-current Δ | Zero-action-history Δ | Cyclic-task-ID Δ |
| --- | ---: | ---: | ---: |
| put_the_cream_cheese_in_the_bowl | 9.04% | 571.31% | 52.13% |
| put_the_bowl_on_the_plate | 6.36% | 548.11% | 68.65% |
| open_the_middle_drawer_of_the_cabinet | 3.36% | 692.36% | 57.76% |
| open_the_top_drawer_and_put_the_bowl_inside | 3.85% | 590.83% | 91.45% |
| push_the_plate_to_the_front_of_the_stove | 3.31% | 962.46% | 110.68% |
| put_the_bowl_on_the_stove | 9.95% | 648.19% | 90.05% |

## Mechanistic reading

- `put_the_cream_cheese_in_the_bowl`: No exact Stage-1 weak baseline exists for this task; the largest offline first-action dependence is `zero_action_history` (+571.31%). Rollouts localize the first coarse bottleneck to `late_approach_or_contact_entry` and the seed success range is 0.400. This supports a bounded mechanism diagnosis, not a single-component causal claim.
- `put_the_bowl_on_the_plate`: Matched clean success increased by 0.267 as an architecture-and-training bundle; the largest offline first-action dependence is `zero_action_history` (+548.11%). Rollouts localize the first coarse bottleneck to `lift_or_transport_transition` and the seed success range is 0.100. This supports a bounded mechanism diagnosis, not a single-component causal claim.
- `open_the_middle_drawer_of_the_cabinet`: Matched clean success increased by 0.200 as an architecture-and-training bundle; the largest offline first-action dependence is `zero_action_history` (+692.36%). Rollouts localize the first coarse bottleneck to `early_grasp_or_articulation_entry` and the seed success range is 0.200. This supports a bounded mechanism diagnosis, not a single-component causal claim.
- `open_the_top_drawer_and_put_the_bowl_inside`: No exact Stage-1 weak baseline exists for this task; the largest offline first-action dependence is `zero_action_history` (+590.83%). Rollouts localize the first coarse bottleneck to `early_grasp_or_articulation_entry` and the seed success range is 0.200. This supports a bounded mechanism diagnosis, not a single-component causal claim.
- `push_the_plate_to_the_front_of_the_stove`: No exact Stage-1 weak baseline exists for this task; the largest offline first-action dependence is `zero_action_history` (+962.46%). Rollouts localize the first coarse bottleneck to `late_approach_or_contact_entry` and the seed success range is 0.900. This supports a bounded mechanism diagnosis, not a single-component causal claim.
- `put_the_bowl_on_the_stove`: No exact Stage-1 weak baseline exists for this task; the largest offline first-action dependence is `zero_action_history` (+648.19%). Rollouts localize the first coarse bottleneck to `no_single_phase_bottleneck_at_gate_resolution` and the seed success range is 0.200. This supports a bounded mechanism diagnosis, not a single-component causal claim.

The matched weak comparison is available only for the two Stage-1 tasks evaluated with the same 10 initial states × 3 seeds and execution horizon 1. Other task improvements or degradations are intentionally left unclaimed. Cross-seed spread and phase reach are rollout evidence; masked-input error deltas are post-hoc localization evidence. Neither identifies a single causal component without retraining or adequately powered rollout ablations.
