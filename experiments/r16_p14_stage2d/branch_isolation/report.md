# Fresh-process branch-isolation gate

Status: **PASS**. Each of 54 smoke branches ran in a unique `spawn` process with a newly created LIBERO environment and newly loaded frozen ACT.

- expected_rows_54: True
- no_missing_or_error_records: True
- unique_spawned_process_per_branch: True
- reconstruction_100_percent: True
- maximum_state_error_le_1e_9: True
- same_action_signature_100_percent: True
- contact_cause_agreement_100_percent: True
- no_S_obs_zero_to_one: True
- branch_order_invariant: True
- actor_inference_side_effect_free: True
