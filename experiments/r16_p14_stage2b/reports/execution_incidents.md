# Execution incidents

## 2026-08-14 — client/network interruption during Phase A

The first two foreground Phase-A workers (actor seeds 7 and 17) were terminated
when the client network/session disconnected. The pre-interruption process had
not written either seed-level atomic shard, so none of those transient rollouts
entered formal evidence and no partial file was treated as complete.

Before restarting, Phase A and Phase B were hardened to write one atomic
per-episode recovery checkpoint. The formal seed shard is still created only
after all required episodes are present and is validated by the aggregate count
contract. The 25-test contract suite was rerun after the change and passed.
Workers were restarted in persistent local `tmux` sessions with at most two
concurrent A800 processes. Recovery checkpoints are operational duplicates and
are ignored by Git after their contents are consolidated into canonical JSONL
shards.

## 2026-08-14 — Phase-A catastrophic-drop monitor repair

The first complete Phase-A aggregate exposed a semantic impossibility in the
stove task: at horizon 8, clean success and `object_drop` were both 58/60, and
all 37 late prefixes that completed the LIBERO task inside the prefix were
marked unfaithful. Code inspection showed that the monitor treated any descent
below the lift threshold as a drop, although the task explicitly requires the
bowl to descend onto the stove and LIBERO's `On` predicate can become true one
simulator step later. This was diagnosed before Phase B or any perturbed outcome
was run.

The monitor now requires both descent below its existing drop-height threshold
and distance outside the preregistered task target region (cream placement
tolerance; 0.12 m stove nominal-future tolerance). A pure contract test covers intended stove
placement versus an off-target drop. The invalidated first aggregate and raw
files were quarantined locally; the canonical Phase-A evidence was regenerated
from the frozen actors and the same seeds/init states. No threshold, task,
actor, horizon, or outcome was selected using downstream results.
