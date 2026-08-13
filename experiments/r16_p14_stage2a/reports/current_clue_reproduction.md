# Current clue reproduction

Status: **PASS**.

The six source rows were parsed mechanically, without a new interpretation:

- M0 safe success: `5/6`.
- M1 safe success: `5/6`.
- M2 safe success: `2/6`.
- M0 failure IDs: `['put_the_cream_cheese_in_the_bowl__demo03__t0078__lead08_shift040mm']`.
- M1 failure IDs: `['put_the_cream_cheese_in_the_bowl__demo06__t0082__lead08_shift040mm']`.
- M0 and M1 failure samples differ: `True`.
- M2-only safe successes when both M0 and M1 fail: `0`.
- M0/M1 per-sample safe union: `6/6`.

**This is calibration-only n=6 hypothesis-generating evidence, not algorithm performance.**

Source SHA256: `d80ce33105729a0902b8e8fc455be83c89800d1648d2c74a044e499ee108fe4c`.
