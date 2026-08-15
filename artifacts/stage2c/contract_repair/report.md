# Stage-2C correctness and contract repair

Status: **PASS** for implementation repair. This does not change the immutable Stage2B replay BLOCKED result.

The only historical failure was `put_the_bowl_on_the_stove__seed29__init10`. The first observable saved-history divergence is global step 181; max history delta is 1.86e-09, while the frozen ACT chunk changes by 6.47e-05.

Fresh reconstructions are mutually byte-exact, so the repair is a pre-outcome 3/3 admission gate plus unstable-event exclusion. No tolerance was loosened.

Replay denominators now include exceptions, and every formal cell requires zero errors. Goal distance is derived from the live BDDL predicate and site/object geometry; demonstration-0 endpoints are absent.

The historical schema did not store per-step MuJoCo/controller/RNG snapshots, so an exact simulator first-divergence step cannot be recovered retrospectively. Stage2C persists those trace hashes prospectively.
