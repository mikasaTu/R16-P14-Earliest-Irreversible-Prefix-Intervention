# Replay evidence index

Formal replay-contract diagnostics are stored at
`artifacts/stage2c/replay_contract_diagnostic.{json,md}`. The raw branch rows
are under `artifacts/stage2c/formal_matrix/`, and the immutable cross-process
probe is under `artifacts/stage2c/pai/replay_probe/`.

The final contract is `BLOCKED`: 363/1,440 recovery prefix cells produced
contradictory pre-operator `S_obs(k)` values, affecting 33/96 events. No raw
row was repaired or discarded.
