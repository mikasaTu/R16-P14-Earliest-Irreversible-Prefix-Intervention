# Frozen perturbation qualification

Selection uses calibration demos 0–9 only: injection-instant violation, delayed nominal target-cause violation, and fresh replay. It never uses proposed-method gain.

| Task | Selected configuration | Qualified configs | Immediate | Delayed nominal | Replay |
| --- | --- | ---: | ---: | ---: | ---: |
| put_the_cream_cheese_in_the_bowl | lead08_shift040mm | 10 | 0.0 | 0.6 | 1.0 |
| put_the_bowl_on_the_plate | none | 0 | None | None | None |
| open_the_middle_drawer_of_the_cabinet | none | 0 | None | None | None |
| open_the_top_drawer_and_put_the_bowl_inside | none | 0 | None | None | None |
| push_the_plate_to_the_front_of_the_stove | none | 0 | None | None | None |
| put_the_bowl_on_the_stove | lead14_lateral050mm | 2 | 0.0 | 0.6 | 1.0 |

A task with no qualifying fixed-grid configuration remains a negative result. Evaluation and held-out splits were not inspected or retuned.
