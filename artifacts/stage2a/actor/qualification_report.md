# History-Conditioned State ACT qualification

Parameter count: `3146503` (limit `10,000,000`). The frozen actor uses four state observations, three executed actions, task ID, and sixteen action queries; it uses no RGB, world model, or perturbation metadata.

| Task | Clean success | Grasp/open | Lift/transport | Pre-release/contact | Smoothness | Policy calls | Qualified |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| put_the_cream_cheese_in_the_bowl | 0.433 | 0.600 | 0.600 | 0.667 | 0.0806 | 265.0 | qualified |
| put_the_bowl_on_the_plate | 0.333 | 0.500 | 0.333 | 0.567 | 0.0740 | 245.5 | qualified |
| open_the_middle_drawer_of_the_cabinet | 0.233 | 0.300 | 0.300 | 0.267 | 0.0445 | 322.2 | not_qualified |
| open_the_top_drawer_and_put_the_bowl_inside | 0.133 | 0.367 | 0.500 | 0.300 | 0.0528 | 563.7 | not_qualified |
| push_the_plate_to_the_front_of_the_stove | 0.400 | 1.000 | 1.000 | 0.500 | 0.0527 | 311.7 | qualified |
| put_the_bowl_on_the_stove | 0.900 | 0.867 | 0.767 | 0.900 | 0.0855 | 138.3 | qualified |

Qualified tasks: `4/6`; actor gate: **PASS**.

Qualification A is clean success in [0.30, 0.90]. Qualification B requires pre-release/contact reach >=0.70 and either no failures or at least half of failures occurring after that phase. No hyperparameter sweep or post-outcome retuning was performed.
