# Stage-2E S1 instrumentation contract (not started)

本文件只冻结未来 S1 的测量合同；S0 是零 rollout、零模型的离线重分析，未创建环境、未加载模型、未提交 PAI，也未启动 S1。

## D1 接触拓扑

每个 simulation step 必须记录规范化、排序后的完整 contact geom-name pair 集合和原始 pair 数据。`contact_count` 不能替代拓扑。记录必须带 `event_instance_id, prefix_k, operator, actor_seed, step` 唯一键。

## D2 可替换 cause 函数

cause 判据必须是离线纯函数，输入为原始接触拓扑、对象/目标几何和 task phase，输出 `(violation, violation_type)`。运行时不得把一个标签写死为唯一判据；既有 release-based label 必须原样保留，新增候选量只能并列记录，不能覆盖既有 label。

## D3 release 前候选观测

S1 必须同时记录一个 release 前可触发的候选量（例如 manipulated object 与非目标 geom 的接触拓扑越界），并明确它与既有 release-based cause 的时间关系、缺失状态和阈值来源。此合同不定义新 idea，也不把候选量当作验证结论。

## D4 fresh-process provenance

每条 branch 记录 `pid`、environment construction hash、actor chunk bytes hash，并沿用 Stage-2D isolation signature 字段：simulator/observation/state-history/action-history/robot qpos-qvel/contact/object/target/gripper/task-phase/CauseTracker/executed-prefix hashes。`pid/env_hash/chunk_hash` 任一缺失时 branch/event 必须 BLOCKED。每个 `(event,prefix,operator,actor_seed,repeat)` 必须是 `multiprocessing.get_context("spawn")` 的独立进程，不能共享 env、wrapper、RNG、cache 或 runtime。

## D5 严格边界

S1 的 actors 固定为 7/17/29；calibration 与 evaluation 严格分离；任何选择器/预算选择必须在 calibration receipt 冻结后才能读取 evaluation 的描述性结果。不得加入 world model、RGB/VLA、训练、actor retraining、learned head 或闭环结果。

`diagnostic_only=true`, `formal_positive_evidence_allowed=false`, `new_idea_generated=false`。
