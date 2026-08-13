# Phase B — Bowl replay reconstruction repair

## Result

Replay gate: **PASS**

| Initializer | Candidate-level insertion pass | All branch-point pass | Contact/outcome match | Maximum final-state error |
| --- | ---: | ---: | ---: | ---: |
| Stage-1 snapshot restore (published) | 86.7% | n/a | n/a | n/a |
| Snapshot restore control, 5 repeats | 86.7% | 81.7% | 82.8% | 3.72 |
| Fresh-env prefix reconstruction, 5 repeats | 100.0% | 100.0% | 100.0% | 0 |

## Contract

Each reconstruction starts from one of five fully independent environment instances, restores the demonstration phase anchor, replays the exact frozen CUDA-reconstructed action chunk, applies the recorded environment perturbation at `d`, and continues to the requested branch prefix. No mutated branch state is shared between repetitions.

The gate requires at least 99% branch-point pass rate, 100% contact/outcome agreement, and final-state max absolute error no greater than 1.0e-09.

## Runtime note

Simulation and reconstruction are CPU-side. A single local A800 is used only for the frozen MLP forward pass because the Stage-1 action hashes were generated with PyTorch 2.5.1+cu124 CUDA; CPU inference produces different floating-point bytes and is rejected by the action-hash contract.

This phase tests branch-state correctness only. It does not establish expert mechanism feasibility or policy performance.
