# Stage-2B A-D checkpoint

A/C/D joint gate: **BLOCKED**; `H_valid=16`.

- Chunk executability: `PASS`
- Event split: `PASS`
- Actor-conditioned perturbation: `BLOCKED`
- Full actor-history replay: `BLOCKED`

The latest user instruction requires Phase E/F to execute even when this checkpoint is blocked. In that case all downstream results are descriptive and cannot receive a positive Track label.
