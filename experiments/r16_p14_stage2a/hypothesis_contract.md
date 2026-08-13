# Stage-2A hypothesis contract

## Immutable boundary

The Stage-1b universal hypothesis remains **killed**. Its frozen decision is
`KILL_CORE_HYPOTHESIS`; Stage-2A cannot delete, overwrite, reinterpret, or
weaken that result. Stage-2A tests two narrower hypotheses and cannot claim
`accepted`, N3/N4 novelty, or a learned R16-P14 method.

## Track A — conditional original mechanism

> In a preregistered deferred-invalidation task family, continuing a
> still-valid nominal prefix and replanning at the last physically safe state
> reduces rework without degrading safety relative to immediate replanning.

The only permitted positive label is `conditional_task_family_support`.

## Track B — policy-relative replanability

> Physical safety and frozen-policy replanability are different. The best
> replanning prefix is the safe state with maximum probability of successful
> replanning, not necessarily the detection state or the last physically safe
> state.

Definitions:

- `d`: perturbation-detection prefix.
- `S(k)`: execution from `d` through `k` has not incurred the explicitly
  registered target-cause violation.
- `C_pi(k)`: safe-success probability when the same frozen actor replans from
  state `s_k`.
- `k_last_safe = max{k : S(k)=1}`.
- `k_best`: lexicographic maximizer of safe-success probability, negative new
  non-nominal actions, then negative policy calls.

Perturbation metadata may be used by the oracle audit and violation labeler,
but never as test-time actor or diagnostic-probe input.

## Bounded novelty label

The working phrase is **Policy-Relative Replanability Window**. It is a bounded
proposal name, not a novelty grade. Two independent primary-literature audits
are required before any novelty assessment, and this checkpoint assigns no N3.
