# Phase F — secondary operator audit

Operator-router result: **NO_OPERATOR_ROUTING_SIGNAL** from 40 events.
Cause-specific local-repair result: **NO_SIGNAL**.
Positive labels permitted: **False**; descriptive router criteria met: **False**; descriptive local-repair criteria met: **False**.

| operator | unique-safe-win events | rate | tasks |
|---|---:|---:|---|
| full_replan | 4 | 0.100 | put_the_bowl_on_the_stove, put_the_cream_cheese_in_the_bowl |
| hold_one_step_replan | 1 | 0.025 | put_the_bowl_on_the_stove |
| bounded_rollback_replan | 0 | 0.000 | none |
| cause_specific_local_repair | 6 | 0.150 | put_the_cream_cheese_in_the_bowl |

Identical budget contract: `True`.
