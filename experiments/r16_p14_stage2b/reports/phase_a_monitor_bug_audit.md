# Phase-A monitor bug audit

This audit was triggered after the first complete 600-episode Phase-A run and
before Phase B or any perturbed/replan outcome was executed.

## Contradiction in the invalidated aggregate

For `put_the_bowl_on_the_stove`:

| horizon | episodes | clean success | object_drop | success AND object_drop | safe_success |
|---:|---:|---:|---:|---:|---:|
| 1 | 60 | 51 | 46 | 45 | 6 |
| 2 | 60 | 57 | 58 | 57 | 0 |
| 4 | 60 | 56 | 56 | 56 | 0 |
| 8 | 60 | 58 | 58 | 58 | 0 |
| 16 | 60 | 57 | 57 | 57 | 0 |

At horizon 8, all 37 late prefixes that completed LIBERO's task inside the
prefix were nevertheless marked unfaithful. At horizon 16 the same was true
for all 44 late anchors. This is incompatible with treating `object_drop` as a
task-specific catastrophic event.

## Root cause and bounded repair

`TaskMonitor.observe` previously marked any post-lift descent below its drop-height
threshold as a drop. However, the BDDL goal is `On(akita_black_bowl_1,
flat_stove_1_cook_region)`: the intended behavior itself requires the bowl to
descend and contact the stove, and `On` may be registered one simulator step
after the height crossing.

The repaired detector preserves the existing height test but additionally
requires the object to be outside the task target region. It uses the existing
cream placement tolerance and the already frozen 0.12 m stove nominal-future
tolerance. No horizon, actor, seed, task, success threshold, or downstream
outcome was changed or read. Test 28 distinguishes intended stove placement
from an off-target descent.

## Invalidated evidence identities

- summary: `e9656286ee711b06c3edb3652203b1bd06093b9da5520cddd0c269eea248a45d`
- raw episodes: `a4ee9ad53bac79daad7bd65bc1d858fcc4343507e9e18f55775c7b874b22b0bf`
- prefix records: `f76e2cd5451c2b9230561494599ff101f0995ff1d9f43d55cefceba02d42bee1`

These first-pass files are quarantined locally and are not used as canonical
evidence. Canonical Phase-A artifacts are regenerated using the same frozen
actors, states, seeds, horizons, and action code with only this monitor repair.
