# Stage-2C release verification

- Formal PAI job: `dlcb8djiituf7gt3` (`Succeeded`, 56,098 seconds)
- Formal event files: 96/96 complete
- Matched-prefix rows: 5,760
- Recovery-operator rows: 17,280
- Formal execution errors: 0
- Replay-invalid events: 33/96
- Stage-2C tests: 39 passed
- Independent release verifier: PASS
- SHA256 manifest entries: 1,352
- `artifacts/stage2c/SHA256SUMS` SHA256:
  `ff2b22e9d264b0b357b50a9774ac4707f162b9b41b27fd8dbae0ef7d4b8404cd`
- GitHub-ineligible files (>=100 MB): 0
- Superseded PAI service records deleted and verified absent: 5/5

The release verifier checks immutable parent identity, actor-event and
qualification counts, formal matrix completeness, the fail-closed decision,
replay diagnostic, PAI lifecycle receipts, exact artifact-manifest coverage,
every artifact digest, and GitHub's per-file size limit.
