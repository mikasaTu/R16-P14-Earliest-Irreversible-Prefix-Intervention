# Stage-2C mechanism reverse explanation

This audit explains the measured increase/decrease using the frozen code paths; it does not generate a new idea.

Paired prefix cells: 1440. Cached-improves: 113; cached-worsens: 60.

Cached and fresh branches differ only in the `k-d` executed prefix. Both receive the same detection-time ACT call, call again at k, and use the same h=16 tail and action budget. Improvement is attributed only when nominal progress/fewer new actions survive without additional cause violations; degradation is tested against stale-prefix violations and cached/fresh action displacement.
