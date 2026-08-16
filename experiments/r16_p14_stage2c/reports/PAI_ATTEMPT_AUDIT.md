# Stage-2C PAI attempt audit

All attempts are preserved rather than hidden. v1/v2 failed sealed preflight before job creation; v3/v4 exposed runtime and working-directory contracts; v5 exposed first-marker and EGL faults; v6 produced the immutable actor-event shards but invalid qualification evidence; the independent replay probe passed; v7 isolated the trace side effect; v8 is the only formal outcome run.

The exact status, JobId, PodUid, duration, failure reason and evidence eligibility of every attempt are recorded in `artifacts/stage2c/pai_attempt_audit.json`. No unrelated PAI job was stopped or modified. At most two A800 GPUs and two GPU workers were used. No actor was trained and no W&B run was created.

v8 (`dlcb8djiituf7gt3`) succeeded after 56,098 seconds and completed 96/96 event files, 5,760 matched rows and 17,280 recovery rows with zero execution errors. Aggregation still failed closed: 33 events produced contradictory pre-operator `S_obs(k)` labels when branches reused one mutable runtime. Instrumentation failures are not scientific negatives, and v8's raw matrices remain diagnostic-only because both the replay contract and perturbation qualification are blocked.

After v8 succeeded, the exact superseded service rows v3–v7 were prepared together, deleted one at a time through the pinned OpenAPI helper, and each received fresh GetJob/ListJobs absence evidence. v8 and the successful independent replay probe were retained; no unrelated job was modified and no CPFS or registry evidence was deleted. Receipts are under `artifacts/stage2c/pai/cleanup/`.
