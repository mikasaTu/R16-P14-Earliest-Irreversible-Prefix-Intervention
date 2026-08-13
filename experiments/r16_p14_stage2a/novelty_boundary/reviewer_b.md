# Independent primary-literature audit — Reviewer B

Audit cutoff: 2026-08-14 UTC. Reviewer B did not read Reviewer A's findings.
The review used primary papers, publisher pages, author project pages, and
official repositories; ambiguous names were preserved as ambiguous.

## Finding

The broad claim “choose an action prefix or replanning horizon according to a
frozen base policy's success/value” is strongly covered. DEHP and BCP optimize
execution horizon around a frozen actor; ACH, ACSAC, and AQC estimate
prefix-relative value; option interruption is the older theoretical ancestor.
CheckVLA also defines an empirical recovery window relative to tested repair
operators.

The bounded distinction left by the search is the conjunction of: a detected
external perturbation; continuing only the same stale nominal chunk; an
independent cause-specific safe set `S(k)`; real branch states; the same frozen
actor's `C_pi(k)`; an interior optimum allowed to differ from both endpoints;
and lexicographic rework/call tie-breaks. This is an operational diagnostic
boundary, not a broad new stopping principle.

Status: `N3 = WITHHELD / NOT_ASSIGNABLE_BY_REVIEWER_B`.

## Primary sources and overlap

| Work | Primary source | Relevant overlap | Missing relative to Stage-2A |
| --- | --- | --- | --- |
| PACE | [Phase-Aware Chunk Execution](https://arxiv.org/abs/2606.00537) | Phase-aware fixed prefix selection | No success-value or independent safety filter |
| DEHP | [Dynamic Execution Horizon Prediction](https://arxiv.org/abs/2606.11408), [project](https://dehp-chunking.github.io/) | Frozen actor and outcome-trained horizon | Decides before execution, no `S(k)` |
| BCP | [Bernoulli-Continuation Policy](https://arxiv.org/abs/2608.03483), [project](https://fleetfootwork.github.io/BCP/) | Ordered continue/replan, success plus call cost | No perturbation-conditioned exhaustive atlas |
| ACH | [Multi-Chunk Q Value Estimation](https://arxiv.org/abs/2605.10044) | Prefix-consistent `Q^pi` ranking | No safe-prefix feasibility constraint |
| ACSAC | [Adaptive Chunk Size Actor-Critic](https://arxiv.org/abs/2605.11009) | State/action prefix Q and joint argmax | Actor/critic co-training; no frozen-policy safety atlas |
| AQC | [Adaptive Q-Chunking](https://arxiv.org/abs/2605.05544) | Per-horizon advantage selection | No cause-specific `S(k)` |
| REMAC | [Masked Action Chunking](https://arxiv.org/abs/2601.20130), [official code](https://github.com/hatchetProject/REMAC) | Prefix preservation and suffix continuity | Handoff timing is not selected by recoverability |
| CheckVLA | [Execution-Time Verification](https://arxiv.org/abs/2607.26789) | Perturbation onset, committed prefix, repair-relative recovery window | First risk crossing rather than success-ranked interior handoff |
| FFDC-WAM | [When to Trust Imagination](https://arxiv.org/abs/2605.06222) | Observation-conditioned continue/replan | No exhaustive branch probability or safety axis |
| Options | [Between MDPs and semi-MDPs](https://www.sciencedirect.com/science/article/pii/S0004370299000521), [TRIO](https://proceedings.mlr.press/v32/mannb14.html), [Option-Critic](https://arxiv.org/abs/1609.05140), [Deliberation Cost](https://ojs.aaai.org/index.php/AAAI/article/view/11831) | Value-relative terminate/switch and decision cost | General abstraction, not a target-cause prefix atlas |

“Adaptive Chunking via State-Action Critic” is ambiguous among ACH, ACSAC,
and AQC. The review does not select one without a title/arXiv identifier.

## Formal collision

For a fixed nominal prefix and continuation policy, a multi-step
`Q^pi(s_d, a_{d:k})` reduces toward continuation value when intermediate reward
is absent and terminal reward is binary success. Thus comparing policy-relative
values of shared prefixes is already prior art. The empirical distinction in
Stage-2A is the external safe-set filter, cause-specific no-violation event,
real post-perturbation branch, and strictly frozen continuation actor.

## Evidence limits

This was a bounded audit, not a systematic review. Multiple closest sources
are new preprints and were not reproduced. CheckVLA's window is repair-set
relative while Stage-2A is frozen-actor relative; the procedures differ, but
the overlap is enough to rule out a broad first-use claim. `C_pi(k)` remains a
finite-rollout empirical estimate and `S(k)` is not a safety certificate.
