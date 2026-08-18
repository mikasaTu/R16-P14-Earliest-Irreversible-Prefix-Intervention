# Stage-2D Phase-3 PAI receipt

- run_id: r16p14-stage2d-phase3-oracle-20260819-v6
- job_id: dlcxqlerueks20h6
- source_commit: 141484c784be92373b3194f3c81c45a8471fa123
- source_tree: 618d2cdcff577ed5505c0a2da285c14593aae4dc
- requested matrix: 154 events x 15 k = 2310 fresh-process oracle branches
- persisted matrix: 2310/2310, error_count=0, missing_shards=[]
- terminal receipt: oracle_appendix/terminal_receipt.json, status=SUCCEEDED
- PAI Job final status: Failed during postprocessing because the first pytest ran before SHA256SUMS was rebuilt after oracle shards were written (test_29); the matrix and terminal receipt were complete and unaffected.
- The repository copy is rebuilt and validated locally after importing the immutable oracle shards.
