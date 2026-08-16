# Stage-2C replay-contract diagnostic

Status: **BLOCKED** (`SHARED_RUNTIME_RESET_INCOMPLETE_OR_ORDER_DEPENDENT`).

All 96 formal events and all planned rows completed. Nevertheless, 363/1440 recovery prefix cells returned both true and false for `S_obs(k)`, affecting 33/96 events.
The same-action `CACHED_MATCHED`/`CACHED_NOQUERY` control disagreed on pre-tail `S_obs(k)` in 62/1440 cells. Independently, 46/192 matched nominal sequences and 217/1152 recovery sequences contained a 0→1 transition.

The recovery code records `prefix_cause_violation` before actor inference and before operator-specific actions. Those rows also execute the same frozen cached prefix. Recovery actor, operator and tail outcome therefore cannot explain the disagreement.

The formal worker reuses one mutable LIBERO environment per event and restores an enumerated state snapshot between branches. The evidence localizes the failure to residual reset/order-dependent runtime state, but does not identify one hidden field strongly enough to claim a narrower root cause.

No raw row was changed or discarded. `INCOMPLETE_PREFIX_GRID` is the fail-closed aggregation label created when contradictory `S_obs(k)` values are set to null; it is not a missing-execution label. All downstream gains/losses remain descriptive only, and no new idea is introduced.
