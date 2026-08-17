# PAI execution contract

Stage-2D is evaluation-only: it reuses three frozen ACT checkpoints and does
not train or save a model. Therefore the training-checkpoint retention and
optimizer autoresume rules are not applicable. Resume is instead fail-closed
at immutable `(event,k,arm,repeat)` JSON shard boundaries; completed shards are
never overwritten.

The formal workflow uses four scientific phases and three sequential, committed barriers on dedicated
`exp-efficiency` with one PAI worker holding two A800s and two concurrent
scientific workers:

1. `phase1`: strict event construction, perturbation qualification, and the
   outcome-blind formal cohort.
2. local mandatory checkpoint barrier: import, report, and commit all Phase 1
   evidence before any atlas branch exists.
3. `atlas`: full calibration atlas only.
4. local barrier: import the atlas, freeze and commit the outcome-blind rule.
5. `phase2`: 3/3 replay admission, nine confirmatory methods, 10,000-draw
   primary statistics, and primary-decision lock.
6. local barrier: import and commit the primary manifest.
7. `phase3`: appendix-only evaluation oracle, mechanism audit, report,
   checksums, and tests.

PAI automatic fault tolerance is disabled on the dedicated pool. The launcher
runs as UID/GID 2254:2254, checks the exact source commit/tree and clean status,
uses only `/mnt/cpfs/zbl-cpfs-new`, and creates no probe job because the local
two-task real-LIBERO smoke already passed.
