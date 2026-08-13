# Independent primary-literature audit — Reviewer A

Audit cutoff: 2026-08-14 UTC. Reviewer A did not read Reviewer B's findings.
Only papers, publisher pages, author project pages, and official repositories
were treated as evidence. Exact-title misses were not treated as proof of
absence.

## Finding

The broad phrase **Policy-Relative Replanability Window** cannot support an N3
claim. Adaptive execution horizons, frozen-policy continue/replan decisions,
prefix-value ranking, online execution verification, repair-relative recovery
windows, and value-relative option interruption are all present in prior work.
The strongest direct collision is CheckVLA Appendix G: it operationalizes an
empirical recovery window relative to the tested suffix-repair set and
explicitly distinguishes that from physical irreversibility.

The only boundary not found as a single prior protocol is narrow: after a
known perturbation, exhaustively branch every stride-1 prefix; separately
measure target-cause safety `S(k)` and frozen-actor recovery `C_pi(k)`; filter
unsafe prefixes; then compare the lexicographic safe-success optimum with the
detection point and last physically safe point. This is best described as a
safety-constrained oracle/evaluation protocol, not a new adaptive replanning
principle.

Status: `N3_WITHHELD_PENDING_DUAL_REVIEW`.

## Primary sources and overlap

| Work | Primary source | Relevant overlap | Missing relative to Stage-2A |
| --- | --- | --- | --- |
| PACE | [Phase-Aware Chunk Execution](https://arxiv.org/abs/2606.00537) | Selects a chunk prefix at a low-speed phase boundary | No perturbation atlas, `S(k)`, or `C_pi(k)` |
| DEHP | [Dynamic Execution Horizon Prediction](https://arxiv.org/abs/2606.11408), [project](https://dehp-chunking.github.io/) | Frozen base policy; return-trained categorical horizon | No independent safety constraint |
| BCP | [Continue or Replan?](https://arxiv.org/abs/2608.03483), [project](https://fleetfootwork.github.io/BCP/) | Ordered prefix continuation; success/call-efficiency objective | Chooses at chunk start; no safe branch atlas |
| AQC | [Adaptive Q-Chunking](https://arxiv.org/abs/2605.05544) | Scores multiple candidate horizons with policy-relative Q/advantage | No perturbation-conditioned safety filter |
| ACH | [Adaptive Action Chunking via Multi-Chunk Q](https://arxiv.org/abs/2605.10044) | Estimates `Q^pi` for all shared prefixes | No separate physical-safety axis |
| REMAC | [Real-Time Robot Execution with Masked Action Chunking](https://arxiv.org/abs/2601.20130), [official code](https://github.com/hatchetProject/REMAC) | Retains committed prefix and regenerates a suffix under latency | Does not optimize the handoff prefix |
| CheckVLA | [Execution-Time Verification with Action-Conditioned World Model](https://arxiv.org/abs/2607.26789) | Online risk crossing; repair-relative empirical recovery window | Does not rank all safe interior handoff points by frozen-actor success |
| When to Trust Imagination | [Adaptive Action Execution for World Action Models](https://arxiv.org/abs/2605.06222) | Uses new observations to continue or replan | No explicit `S(k)` or exhaustive prefix oracle |
| Options | [Sutton, Precup & Singh](https://www.sciencedirect.com/science/article/pii/S0004370299000521), [TRIO](https://proceedings.mlr.press/v32/mannb14.html), [Option-Critic](https://arxiv.org/abs/1609.05140) | Continue/terminate/interruption is policy-value relative | General framework; no cause-specific chunk atlas |

“Adaptive Chunking via State-Action Critic” is not a uniquely verifiable
title; AQC and ACH are distinct plausible referents. “REMAC” is also
ambiguous unless its full title is frozen. Neither ambiguity was silently
resolved.

## Evidence limits

Several closest papers are recent preprints, and their headline results were
not independently reproduced in this audit. Official runnable code was
confirmed only for masked-action-chunking REMAC. The exact phrase not appearing
in bounded search does not establish conceptual novelty. `S(k)` is only an
empirical, target-cause-specific predicate, not a general safety guarantee.
