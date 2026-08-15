# Stage-2C Phase A/B checkpoint

Checkpoint frozen after contract repair, actor-event admission, perturbation qualification, and the two-family integration smoke. The user explicitly required all downstream experiments to continue even when a gate failed, so the blocked qualification is preserved and the later matrix is diagnostic-only.

## Source identities

- Immutable parent commit: `6eae66d23313cc97231249bfa1c40dc1767ea727`
- Immutable parent tree: `ddcedd60f4f4e2878f8a4400d65e9e888f00cdd1`
- Formal PAI source commit: `55c180bfd1abb9767e146cbd7ec5554e6760a0d5`
- Formal PAI source tree: `a1e2caee307b2270bc167f26164d473e7908baf0`
- Pre-checkpoint delivery HEAD: `219aba2d035dd035ac32b4fe8b7d156c0d1629bf`
- Pre-checkpoint delivery tree: `4702b920b9358e0e7b275dd5f2ea6775e43f272e`
- Formal job: `dlcb8djiituf7gt3` (`r16p14-stage2c-formal-20260816-v8`)
- Hash-locked reused event shards: 240 files, SHA256 `4fe57054d32c439201b81da0e433c9c0566c3222b21e28f994f584724474697c`

## Phase A — repaired contracts

- Contract repair: **PASS**.
- Replay denominator now counts every attempted reconstruction; an error blocks the cell.
- Corrected historical Stage-2B stove replay rate: `23/24 = 0.958333` for each audited severity, with `error_count=1`; the legacy error-dropping denominator incorrectly reported `1.0`.
- Stage-2B descriptive prefix grid contains 0 observed `0→1` safety transitions across 34 event/severity cells. Stage-2C independently requires a complete 15-prefix grid and rejects `NONMONOTONIC_CAUSE_PREFIX` events.
- Goal distance is read from live BDDL state: `ObjectState.check_ontop` for object targets; live site position/rotation/size for stove, plate and drawer regions. No demonstration-0 endpoint is used.
- The sole Stage-2B replay failure remains classified as `LONG_HORIZON_NUMERICAL_CONTEXT_DRIFT_AMPLIFIED_BY_ACT`: first observable difference at global step 181, state-history max difference `1.862645149230957e-09`, simulator-state max difference `5.0034199006177005e-11`, and ACT chunk max difference `6.473064422607422e-05`. This does not change the immutable Stage-2B `BLOCKED` result.
- A later Stage-2C infrastructure audit found that diagnostic trace collection called `current_observation`, which refreshes LIBERO's observable cache. Formal source `55c180b` makes that refresh schedule identical whether trace rows are retained or discarded. Exact replay gates were not relaxed.

## Actor-event admission

- Attempts: 240.
- Admitted frozen-ACT events: 238.
- Ineligible events: 2 (`0.8333%`), both stove events.
- Replay-unstable exclusions after the trace fix: 0 (`0%`).
- Every admitted event passed 3/3 fresh reconstruction, exact anchor state/history, exact original chunk hash, branch-order invariance, and the no-error gate before intervention outcomes were read.

## Phase B — qualification result

- Completed attempts: `891/891`; missing: 0; replay errors: 0.
- Qualification status: **BLOCKED**.
- Frozen failure label: `BLOCKED_BY_SECOND_FAILURE_FAMILY`.
- Target shift (`put_the_cream_cheese_in_the_bowl`): 0 qualifying severities. The diagnostic continuation uses `shift_040mm` and `shift_060mm`; both have delayed cause-violation rate 0.
- Candidate path task 1 (`put_the_bowl_on_the_stove`): 0 qualifying severities. The diagnostic continuation uses `future_14_lateral_020mm` and `future_06_lateral_040mm`; the best delayed rate is `0.0909`, and the strict no-contact-at-injection cell gate fails.
- Candidate path task 2 (`push_the_plate_to_the_front_of_the_stove`): 0 qualifying severities; best delayed rate 0 and no fully injection-contact-free cell.
- Candidate path task 3 (`open_the_top_drawer_and_put_the_bowl_inside`): 0 qualifying severities; 3 cells are injection-contact-free, but the best delayed rate is only `0.0714`.
- No task, severity, threshold, or baseline was selected from Track-A/Track-B outcomes.

## Frozen diagnostic event pool and smoke

- Pool: 96 event instances = 2 tasks × 2 splits × 24 events.
- Per task/split: 8 events from each generator seed and 12 from each of two frozen diagnostic severities.
- Minimum-data shape gate: PASS, but it does not repair the upstream perturbation-family failure.
- Integration smoke: 2/2 complete event files, 16 matched rows, 48 recovery rows, 0 errors.

## Exact continuation

The formal job was already running when this checkpoint was committed, because the user's latest instruction overrides the preregistered early stop while forbidding positive labels after a failed gate.

```bash
cd /mnt/cpfs/zbl-cpfs-new/USERS/leon/code/pai-job-registry
./bin/pai-job get dlcb8djiituf7gt3 --run-id r16p14-stage2c-formal-20260816-v8
```

The immutable result constraints remain: Stage-1b `KILLED_IMMUTABLE`; Stage-2B `BLOCKED`; Track A/B cannot receive a positive label after the failed family gate; local repair and operator router remain retired; `accepted=false`; novelty remains no higher than N2.
