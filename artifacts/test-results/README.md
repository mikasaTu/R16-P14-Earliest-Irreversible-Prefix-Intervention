# Publication-tree verification results

The repository was reverified before its initial GitHub publication.

## Evidence integrity validator

- Result: PASS
- Clean rollout rows: 360
- Oracle candidate rows: 90
- Complete final checkpoints: 9
- Instrumentation parity records: 3
- Recomputed formal artifact/config/manifest hashes: 8
- Persisted output: `release_verification.json`
- Full artifact manifest: 84/84 SHA-256 entries passed

## Pytest and simulator replay

- Result: 5 passed, 0 failed, 0 errors, 0 skipped
- Duration: 87.469 seconds
- JUnit output: `pytest.xml`
- Python: 3.11.11
- PyTorch: 2.12.1+cu130
- CUDA runtime reported by PyTorch: 13.0
- MuJoCo: 3.6.0
- robosuite: 1.4.0
- Rendering: headless EGL, `CUDA_VISIBLE_DEVICES=0`

The five tests cover aggregation/gate decisions, action-chunk episode boundaries, incomplete-checkpoint rejection, model tensor shape, and deterministic MuJoCo snapshot/suffix replay.

The publication test environment is intentionally distinguished from the formal PAI environment (PyTorch 2.5.1+cu124). The simulator test passed in both the development validation chain and this final publication-tree rerun.

## Static checks

- `python -m compileall` passed for the experiment package and release validator.
- `bash -n` passed for the exact PAI launcher and the convenience command wrapper.
- The release validator independently recomputed each final checkpoint SHA-256 rather than loading unchecked pickle data.
