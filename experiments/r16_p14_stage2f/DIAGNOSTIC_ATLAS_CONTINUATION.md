# Phase2 用户要求继续执行的诊断性路径

本文件是在 Phase1 已产生部分 calibration 结果、但尚未读取任何 evaluation 原始 outcome/trace 及尚未运行 Phase2 的时点写入。它是执行范围修正，不是事前预注册。

用户正文要求“必须完成计划里面的全部实验，不能因为一个gate不到就停止验证其他实验”。此前 EXECUTION_CONTINUATION_AVAILABLE_CALIBRATION.md 对短样本后的所有 evaluation 全面 open-deny，超出了该正文所要求的继续执行范围。本补充保留原有正式路径与失败结论，为继续完成原计划的算子族、参考臂与噪声地板测量建立独立诊断性路径。

## 保留的正式结论

- PREREG_S1.md 与 K1/K2/K3 数值不修改。
- 正式 Phase1 selection_receipt.json 保留 BLOCKED/null；样本短缺与结构不可行组合原样记录。
- 正式 atlas 入口仍要求原有 SELECTED、完整计划样本、K2数值合格及已提交的 Git 证明；不放宽该入口。
- 诊断性继续执行不代表 K1/K2 通过，不能把缺失事件补成已执行，也不能把所得 K3 数字视为原计划前提完备的确认性结论。

## 单独选择与 evaluation 解封顺序

1. 等待全部19440个现有 Phase1 请求落盘，完成源事件、分片、参考臂、运行身份和未知错误检查。真实 horizon 不可行请求保留 BLOCKED；未知错误或缺失分片不能作为继续依据。
2. 只读取 calibration 的九档完整事件支持集统计。优先在两个任务同时满足原K2数值条件的档中，按原顺序：min-task oracle降序、min-task gap降序、action_budget升序、tail_horizon升序，取首档。
3. 若九档均不满足原K2数值条件，记录该失败；为执行用户要求的后续测量，在全部有效档中按同一排序取首档。该 fallback 是本文件声明的诊断性执行规则，不是“通过K2”或重新寻找有利阈值。
4. 单独写入 phase1/diagnostic_selection_receipt.json：status=DIAGNOSTIC_SELECTED、selection_source=calibration_only、confirmatory=false、缺失前提、全部候选及排序、选定预算、输入SHA和本文件SHA。
5. 将该 receipt 提交并推送到 GitHub main，回读一致，生成其真实 Git 对象成员证明和独立 authorization。只有验证这些证据后，诊断性 atlas 入口才可读取 evaluation。正式 receipt 不被覆盖。
6. 预算一经提交不因 evaluation 结果改变。不增加初始状态、不换split、不使用reserve补样、不训练actor。

## 实际执行与报告

按原 Phase2 定义执行两个任务全部现有 calibration 与 evaluation 自然失败事件，3个恢复seed、固定四算子、k={2,4,6,8,10,12,14,16}以及4个参考臂；复用干预和运行绑定完全相同的 Phase1 calibration 分支，补齐其他请求。所有结果标注 diagnostic_continuation 和所用 selection SHA；保留新运行与复用运行的源身份。

按原定义计算两族及U_full边界、双方safe success、crossing、Spearman、所有seed split-half噪声地板和10000次(task,init) cluster bootstrap（seed216214）。结构不可行组合按可验证的源anchor与固定horizon处理，报告完整支持与排除原因；不把缺失/BLOCKED当False或None边界。

K1/K2的正式失败、K3各数值门槛是否满足、诊断性支持集大小与推断限制分别报告。若样本或边界数不足，明确数值门槛未满足与证据不足，不将其偷换为已证明“相对性平凡”。不启动S2或新idea。

总资源仍为最多2个PAI作业、每个2×A800、20GPU小时上限；禁跑时间控制和自动恢复规则保持。预算上限或真实契约违规仍可终止执行，并如实报告未完成部分。
