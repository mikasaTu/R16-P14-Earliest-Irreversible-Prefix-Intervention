# PAI execution provenance

This directory preserves the credential-free launch contract for the formal LIBERO Stage-1 pilot.

- `launchers/r16p14_libero_stage1.sh` is the exact final command payload. Its SHA-256 is `3c4c83dc5c02b5b9ef780530589e22c4b451a2ab446f043112b224c2c715990a`.
- `templates/r16p14-libero-stage1-exp-efficiency-2gpu.json` records the resource, mount, fault-tolerance, identity, provenance, and evidence contract.
- `successful_job_summary.json` is the read-back summary of successful job `dlc1l9akne34qq7k`.
- `recovery_audit.json` records two sealed pre-create failures, three terminal failed jobs that stopped before first real work, and the final successful recovery. All failed jobs had terminal failed pods; the cleanup target count is zero.

The launcher expects credential values to be injected at runtime and only checks that required variables exist. No token, access key, SSH private key, W&B key, PAI credential file, controller nonce, or controller state is committed.

The CPFS paths and internal resource/data-source identifiers are retained because they are part of the exact reproducibility contract. The template should be adapted and revalidated before use outside the original PAI workspace.
