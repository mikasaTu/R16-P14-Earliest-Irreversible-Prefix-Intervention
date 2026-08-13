# Fresh-environment replay gate

Every branch-point reconstruction uses an independent environment, restores the demonstration anchor, replays the exact nominal prefix, injects the same environment-only perturbation, and then reaches the branch point. Snapshot restore is not the primary initializer.

| Anchor | Task | Branch points | Reconstruction passes | Max state error |
| --- | --- | ---: | ---: | ---: |
| positive | put_the_cream_cheese_in_the_bowl | 3 | 3/3 | 0.0 |
| negative | open_the_middle_drawer_of_the_cabinet | 3 | 3/3 | 0.0 |

Overall status: **PASS**.
