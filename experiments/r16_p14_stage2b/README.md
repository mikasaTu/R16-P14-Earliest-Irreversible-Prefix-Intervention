# R16-P14 Stage-2B

Action-chunk faithfulness and actor-conditioned replanability pilot using the
three frozen Stage-2A HistoryConditionedStateACT checkpoints. No actor is
retrained and no learned head/router is introduced.

Canonical raw evidence is written under `artifacts/stage2b/`; human-readable
reports and the immutable decision live in this directory. Every long-running
command writes atomic per-seed or per-event completion files and resumes only
missing units.

## Reproduction order

Run from the repository root with at most two concurrent GPU workers:

```bash
./experiments/r16_p14_stage2b/commands.sh freeze
./experiments/r16_p14_stage2b/commands.sh chunk-seed 7 cuda:0
./experiments/r16_p14_stage2b/commands.sh chunk-seed 17 cuda:0
./experiments/r16_p14_stage2b/commands.sh chunk-seed 29 cuda:0
./experiments/r16_p14_stage2b/commands.sh chunk-aggregate
./experiments/r16_p14_stage2b/commands.sh events-seed 7 cuda:0
./experiments/r16_p14_stage2b/commands.sh events-seed 17 cuda:0
./experiments/r16_p14_stage2b/commands.sh events-seed 29 cuda:0
./experiments/r16_p14_stage2b/commands.sh events-aggregate
./experiments/r16_p14_stage2b/commands.sh perturbations cuda:0
./experiments/r16_p14_stage2b/commands.sh replay cuda:0
./experiments/r16_p14_stage2b/commands.sh checkpoint-report
./experiments/r16_p14_stage2b/commands.sh atlas-smoke cuda:0
./experiments/r16_p14_stage2b/commands.sh atlas-seed 7 cuda:0
./experiments/r16_p14_stage2b/commands.sh atlas-seed 17 cuda:0
./experiments/r16_p14_stage2b/commands.sh atlas-seed 29 cuda:0
./experiments/r16_p14_stage2b/commands.sh atlas-aggregate
./experiments/r16_p14_stage2b/commands.sh operator-seed 7 cuda:0
./experiments/r16_p14_stage2b/commands.sh operator-seed 17 cuda:0
./experiments/r16_p14_stage2b/commands.sh operator-seed 29 cuda:0
./experiments/r16_p14_stage2b/commands.sh operator-aggregate
./experiments/r16_p14_stage2b/commands.sh mechanism-audit cuda:0
./experiments/r16_p14_stage2b/commands.sh test
./experiments/r16_p14_stage2b/commands.sh report
./experiments/r16_p14_stage2b/commands.sh checksums
```

The latest user instruction overrides only early stopping: A failed A-D gate
does not cancel later runs, but it prevents a downstream positive label. If no
valid `H_valid` exists, the frozen diagnostic continuation horizon is 8.

Feishu nodes:

- step4: https://icnbwz7kd1ui.feishu.cn/wiki/GQ1tw5h8qicaQAkOAPncNVxSn6e
- experiment report: https://icnbwz7kd1ui.feishu.cn/wiki/DiaGwwxjFiZ4p2k7jzqcrSy9nvd
