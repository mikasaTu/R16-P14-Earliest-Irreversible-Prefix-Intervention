# Phase C — expert action-chunk feasibility gate

## Decision

**KILL_CORE_HYPOTHESIS**

The gate became impossible at calibration: only one of three tasks produced any perturbation satisfying all frozen constraints. Evaluation demos 10–39 and held-out demos 40–49 were not inspected. Continuing would require retuning failed tasks on non-calibration data, which is forbidden.

## Task result

| Task | Calibration result | Selected configuration | Median window | Median post-detection retention |
| --- | --- | --- | ---: | ---: |
| open_the_middle_drawer_of_the_cabinet | no_qualifying_configuration | none | n/a | n/a |
| put_the_bowl_on_the_plate | no_qualifying_configuration | none | n/a | n/a |
| put_the_cream_cheese_in_the_bowl | qualified | lead08_shift040mm | 7.0 | 50.0% |

Bowl never satisfied all phase/immediate-contact/rate constraints. Drawer had one rate-qualified near miss (`offset+00_clear035mm`) but its actual median recoverable window was 0. The single bounded replacement selected `lead08_shift040mm`: 0% immediate violations, 60% delayed violations, median window 7, and median retention 50% on calibration demos.

## Selected replacement M0/M1/M2 (calibration only)

| Method | Cause violations | Safe successes | Median new non-nominal actions |
| --- | ---: | ---: | ---: |
| M0 | 1/6 | 5/6 | 34.0 |
| M1 | 1/6 | 5/6 | 33.5 |
| M2 | 3/6 | 2/6 | 7.0 |

The timing comparison is nonempty but has no aggregate safety advantage: M0 and M1 each have 1/6 cause violations and 5/6 safe successes. M2 executes far fewer new actions but is less safe (3/6 cause violations; 2/6 safe successes), so there is no operator win.

Independent calibration prefix replay passed 885/885 branch points (100.0%).

No BC policy, learned risk model, RGB model, world model, policy training, Phase D, or Phase E was executed.
