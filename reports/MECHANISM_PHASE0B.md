# Phase0B 自然事件产出机制分析

本报告采用 code-first 的证据流程：先读实际 collector/runtime/TaskSpec，再用已有 calibration/infrastructure episode 元数据和限定的四条 calibration trace 复核；没有提出新 idea、没有启动 GPU、没有读取 sealed evaluation 或 evaluation trace/outcome。结论区分代码保证与有限样本观测。

## 结论

K1 产出不足的主过滤环节是“clean failure 到达任务 horizon 但没有形成 structural anchor”。calibration 中 cream 为 48/120，bowl 为 58/120；这两组样本全部在各自固定 horizon 结束。clean success 也不进入 qualified natural failure：cream 53/120、bowl 51/120。最终 qualified natural failure 只有 cream 19/120、bowl 11/120，分别距每 task 的 25 条门槛少 6、14 条。

collector 的代码保证是：先在 collector.py:50-57 检查 grasp、连续两次 lift、前一夹爪值为正且当前未成功，才创建 event；循环成功后在 collector.py:82-85 立即停止并用 qualified = event is not None and not success 定义自然失败。因此没有 anchor 的失败不会进入 qualified，已成功的 episode 也不会进入 qualified。

## 具体分母与 actor seed

| task | calibration n | clean success | failure 无 anchor | qualified natural failure | anchor failure 非 qualified |
|---|---:|---:|---:|---:|---:|
| put_the_bowl_on_the_plate | 120 | 51 | 58 | 11 | 0 |
| put_the_cream_cheese_in_the_bowl | 120 | 53 | 48 | 19 | 0 |

每个 task 的 calibration 分母是 120；每个 actor seed 是 40 条。括号内为 qualified / 40：

| task | actor 7 | actor 17 | actor 29 |
|---|---:|---:|---:|
| put_the_bowl_on_the_plate | 2 / 40 (5.0%) | 4 / 40 (10.0%) | 5 / 40 (12.5%) |
| put_the_cream_cheese_in_the_bowl | 5 / 40 (12.5%) | 3 / 40 (7.5%) | 11 / 40 (27.5%) |

完整 per-seed 分解在 mechanism_yield.json；例如 cream 的 actor 7/17/29 分别为 success/no-anchor/qualified = 5/30/5、27/10/3、21/8/11，bowl 分别为 20/18/2、12/24/4、19/16/5。这里是描述性计数，不是 actor 性能因果结论。

## K1 计数

| task | calibration qualified / n | evaluation qualified / n | calibration deficit | evaluation deficit | K1 |
|---|---:|---:|---:|---:|---|
| put_the_bowl_on_the_plate | 11 / 120 | 13 / 120 | 14 | 12 | FAIL |
| put_the_cream_cheese_in_the_bowl | 19 / 120 | 23 / 120 | 6 | 2 | FAIL |

代码中的 K1 判定在 collector.py:155-159，要求 calibration 和 evaluation 两个 split 各至少 25 条。这里 qualification metadata 的全 split 计数只用于计数，不读取 evaluation trace/outcome。

## 标签与机制边界

release-outside-target 和 prerelease topology 是 collector 在 collector.py:88-89 生成的后验诊断，不能解释 qualified 数量的主过滤。calibration qualified 样本中，cream 只有 5/19 有 release violation、1/19 有 topology candidate；bowl 为 3/11、6/11，且两类标签与 qualified 并非等价条件。runtime 的 execute_branch 在 runtime.py:484-503 接收已有 event，safe_success 在 runtime.py:411-465 属于 recovery 分支，因此不能把 recovery label 倒推为 Phase0B 产出过滤。

## 真实样本与 trace 证据

以下四条 trace 均来自 calibration，且已核对 episode 记录的 trace SHA 与文件 SHA；其余 trace 不纳入本报告。

- put_the_cream_cheese_in_the_bowl/init023__actor7.json — qualified；steps=360，anchor=131，records=5954，trace SHA256 0a5bab413b6688458df854bbeb3988f5958982b309bf18e68f3722ac5274af5e。release violation=False，topology candidate=True。
  - event provenance: checkpoint 821177a82cc470e108082fd3c0f6913983236a2fdf142de2fe51fc37c44240ca，init-pool a41be3638b78409a15646ebd2f81c818a7773937c93951baebd25c6621d66b69，init state bd141f60d0f58925f4e1e55b1b15e817cd5051f24e6193d4aa0ceeefe9e8e662，actor run stage2a_shared_multitask_seed_7。
- put_the_cream_cheese_in_the_bowl/init031__actor17.json — clean success；steps=89，anchor=None，records=650，trace SHA256 d36620094041e9977042aa758edb9f64e447dc56a8bf7cf358e014bd11ae0aa0。release violation=False，topology candidate=False。
- put_the_cream_cheese_in_the_bowl/init011__actor7.json — failure without anchor；steps=360，anchor=None，records=9360，trace SHA256 474618350b1a6eda082fb982dfded7941aa4d57c41bf2ce30b6c176961ab3b60。release violation=False，topology candidate=False。
- put_the_bowl_on_the_plate/init017__actor29.json — qualified；steps=320，anchor=146，records=4524，trace SHA256 3a387fb3afa1efb5ab099ff2897d7b0a13bb7680b3cd91850e995938b3fffa51。release violation=True，topology candidate=True。
  - event provenance: checkpoint 0cf34a3e535525345306a2b322aae3b1bd6ebd6cd71dc653e2a91393e2b79d1a，init-pool bc30ad0916e950b3fed126a03d36f552484761f4ef8d1f9269c1ed21fe177e2b，init state deac07ec613e070bf70500528854d3f91494908c88acec6df8146db0a5ca80e6，actor run stage2a_shared_multitask_seed_29。

cream init023/actor7 展示了 anchor 但未 release violation 也可 qualified；cream init011/actor7 展示了完整 horizon 但无 anchor，且其 ignored pre-lift release 不会被误标为 violation；bowl init017/actor29 展示了 anchor、topology candidate 和 release violation 同时出现。

## lineage 与限制

Calibration rows 的 source_commit 全部为 6b788a0764904e11e022c4330a74fa3e009c9a33，runtime receipt 全部为 3d7b252885b4047fb93140c305a3ef4a1d3ec9aa450d90341befd2b7e26a5e44。qualified event 在两个 task 中对同一 actor seed 使用同一 checkpoint：seed 7=821177…21604，seed 17=83ee61…21604，seed 29=0cf34a…79d1a；task 各自 init pool hash 保持 task 内固定。该 provenance 支持“同一固定 actor/checkpoint/runtime 下的产出统计”，不证明 actor 被重新训练，也不把旧 infrastructure episode 与新正式 episode 混作性能改进。

对 no-anchor failures，代码和 episode 元数据只能保证“在 fixed horizon 内没有满足 event 条件”；由于没有 anchor/event，本分析无法把缺失归因到 grasp、两步 lift、previous gripper 或其他单一条件，细分保持 unknown。

机器可读计数、输入清单 digest、source hashes、四条 trace provenance 与上述限制见 artifacts/stage2f/phase0b/mechanism_yield.json。
