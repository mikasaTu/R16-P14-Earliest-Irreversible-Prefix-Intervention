# Stage-2C PAI attempt audit

All attempts are preserved rather than hidden. v1/v2 failed sealed preflight before job creation; v3/v4 exposed runtime and working-directory contracts; v5 exposed first-marker and EGL faults; v6 produced the immutable actor-event shards but invalid qualification evidence; the independent replay probe passed; v7 isolated the trace side effect; v8 is the only formal outcome run.

The exact status, JobId, PodUid, duration, failure reason and evidence eligibility of every attempt are recorded in `artifacts/stage2c/pai_attempt_audit.json`. No unrelated PAI job was stopped or modified. At most two A800 GPUs and two GPU workers were used. No actor was trained and no W&B run was created.

Instrumentation failures are not scientific negatives. Only v8's completed, error-free matrices may enter Stage-2C aggregation, and even those remain diagnostic-only because perturbation qualification is blocked.
