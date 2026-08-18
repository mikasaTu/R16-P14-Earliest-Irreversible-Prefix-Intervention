# Stage-2D Phase-2 PAI run history

This directory preserves the complete control-plane history for the
confirmatory evaluation attempts.  Failed and stopped attempts are retained;
they are not scientific replicates and are not silently removed from the
record.

| attempt | run | JobId | terminal status | source | outcome |
| --- | --- | --- | --- | --- | --- |
| v1 | `r16p14-stage2d-phase2-20260819-v1` | `dlc1tvopvumzq6j0` | Failed | c6ddbda / fa501143 | dirty source guard; local smoke shard made the source tree non-clean |
| v2 | `r16p14-stage2d-phase2-20260819-v2` | `dlc16untoqnvhie4` | Failed | c6ddbda / fa501143 | launcher nested-heredoc syntax error at line 124 |
| v3 | `r16p14-stage2d-phase2-20260819-v3` | `dlcpneeiyv92xcl3` | Failed | c6ddbda / fa501143 | unclosed `write_marker` function/heredoc syntax |
| v4 | `r16p14-stage2d-phase2-20260819-v4` | `dlc1c4lq8alldb0k` | Stopped | c6ddbda / fa501143 | marker write was a no-op; 188 completed replay shards were frozen and imported, without deleting evidence |
| v5 | `r16p14-stage2d-phase2-resume-20260819-v5` | `dlc1vk1qfeg3kzvw` | Succeeded | 4f3974a / e879958 | resumed/imported v4 shards; 462 replay + 1386 method branches, zero errors |

The v5 source differs from v4 only in launcher/control-plane and tests.  The
posthoc science-equivalence receipt records byte-identical Stage-2D scientific
modules, preregistration, metric contract, and frozen-rule manifest.  The
formal scientific status remains diagnostic-only because the upstream event
construction and perturbation qualification gates are blocked.

The v4 recovery receipt and shard list are kept under `v4/`; v5 runtime
receipts, including the import and terminal barriers, are under
`v5/runtime_receipts/`.
