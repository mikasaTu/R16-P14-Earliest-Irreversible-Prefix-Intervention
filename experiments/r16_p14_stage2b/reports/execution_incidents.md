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
