# 现有 calibration 样本的独立网格执行记录

本说明在任何 Phase1 恢复分支之前封存。正式采集源为6b788a0764904e11e022c4330a74fa3e009c9a33。cream calibration 120条已完成，仅19个合格自然失败事件；bowl采集继续。

用户任务正文明确要求“必须完成计划里面的全部实验，不能因为一个gate不到就停止验证其他实验”。据此，继续执行现有 calibration 事件上的独立 configured grid 与参考臂；不把 K1 失败作为取消这些可执行组合的理由。

- PREREG_S1.md、既有执行补充和K1/K2/K3数值均不改写；原计划每任务20事件的要求保留。
- 对每任务按原顺序取前min(20,实际合格calibration数)个事件，执行全部9预算、3恢复seed、4算子、5前缀及4参考臂。缺少的事件不补样、不复制、不插值、不换split或使用reserve/evaluation。
- 结果记录planned_events=20、实际事件数、missing_planned_events与planned_sample_complete。COMPLETE_AVAILABLE_REQUESTS_SAMPLE_SHORTFALL仅表示现有请求执行完，绝不代表计划样本完整。
- 统计可报告现有观测下每个预算的oracle、weakest和gap，并同时报告分母与缺失；这些数值不能替代完整Phase1 K2判决。
- 任一任务不足20个计划calibration事件时，selection receipt不得为SELECTED，evaluation继续open-deny。即使部分网格数值落入K2区间也不形成可发布的选择。
- 若有horizon不可行组合，保留BLOCKED并继续其他独立组合；不把它当失败概率样本。
- 不改变算子、actor、初始池、数据权限、20GPU小时资源上限；不提出新idea，不启动S2。
