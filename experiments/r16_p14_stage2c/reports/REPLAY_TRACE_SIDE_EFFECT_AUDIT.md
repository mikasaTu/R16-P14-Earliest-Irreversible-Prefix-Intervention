# Replay trace side-effect audit

This audit explains the Stage-2C v6/v7 cross-process replay failure. It is separate from the immutable Stage-2B numerical-drift failure.

The supposedly diagnostic path `runtime_trace_row()` calls `current_observation(env)`. In LIBERO that call refreshes the observable cache. Event construction and `capture_trace=true` reconstruction therefore refreshed the cache at the initial state and after every replayed action, while the old `capture_trace=false` path skipped those calls. Trace collection was not observationally inert.

A controlled legacy-path reproduction first diverged at global step 1 in `state_history_hash`. The restored anchor simulator state and three-action history remained exact, but the four-state history reached max absolute error `0.0956674814` across 276 values; the frozen ACT chunk then changed by up to `0.0226997137` across 96 values.

Commit `55c180b` always executes the refresh-bearing trace computation and lets `capture_trace` control only row retention. A dev14 `true/false/false/true` sequence passed all four exact checks in every run. The separate PAI job `dlckxv47eepcwle3` also reconstructed the same v6 event in three fresh Python processes; all three passed 4/4 exact checks with zero state/chunk delta.

This is an instrumentation repair, not a policy improvement and not a new research idea. No exactness tolerance was relaxed, and no immutable Stage-1b or Stage-2B conclusion changes.
