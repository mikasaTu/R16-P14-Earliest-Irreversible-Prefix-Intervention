# R16-P14 Stage-2A First Checkpoint 实验报告

## Checkpoint 结论

**`FIRST_CHECKPOINT_PASS_READY_FOR_BOUNDED_ATLAS`**

本检查点已完成计划要求的 Phase 0 线索复现、六任务结构筛选、固定扰动网格校准、ACT 训练 smoke、三 seed clean/phase gate，以及正负锚点 fresh-environment replay gate。Actor gate 为 PASS，replay gate 为 PASS；但扰动资格仅有 2/6 任务通过，因此后续 Atlas 必须限制在这两个任务，不能用 evaluation/held-out 结果返调另外四个负任务。完整 Atlas 尚未启动，本报告不对 Track A、Track B 或最终 overall 作结论。

## 不可变边界与精确快照

- 本 checkpoint 代码与原始证据 commit：`df20e31cefc4db22ba23f2b61284469e781d5558`。
- 对应 Git tree：`bfe056abce8550cfc3ece69cfd3198f690728dfa`。
- 分支：`agent/stage2a-safe-replanability-atlas`；remote：`git@github.com:mikasaTu/R16-P14-Earliest-Irreversible-Prefix-Intervention.git`。
- 冻结 Stage-1 commit/tree：`a1b61194a8382f5b1a247b9cd9b140645ff2aeb8` / `53001c43fbbb165c0a1f2c71f9cbd4c81b9d0ced`。
- 冻结 Stage-1b commit/tree：`e29e3ead42fd1799b412a4968e6a67aac3784874` / `4016b05942e6dfb291f1bc3a2644e177a208b608`。
- Stage-1b 决策保持 **`KILL_CORE_HYPOTHESIS` / `KILLED_IMMUTABLE`**，本阶段不能撤销。
- Stage-1b REPORT SHA256：`17d9d8a5deb11aa9e070168920455c9fbdd3129a96ee695336f12332160196fc`；decision SHA256：`3d05791be98d6ee83858e1728b438cd4a408c27e4aae2556f649ab07175d9984`。
- `accepted=false`；未训练 learned irreversibility/replanability head；未使用 π0.5、RGB、DINO-WM 或 world model。

说明：报告自身在上述证据 commit 之后生成，因此交付文档 commit 会不同；上述 commit/tree 是包含全部实验代码、权重和原始结果的可复现实验快照。

## Phase 0 — 当前线索机械复现

从冻结的 `selected_config_paired_metrics.csv` 得到 M0=5/6、M1=5/6、M2=2/6；M0 与 M1 的失败样本不同，二者逐样本 safe union 为 6/6；当 M0、M1 同时失败时，M2 独有 safe success 为 0。

**这是 calibration-only、n=6、hypothesis-generating 证据，不是算法性能。**

输入 SHA256：`d80ce33105729a0902b8e8fc455be83c89800d1648d2c74a044e499ee108fe4c`。

## Phase 2 — 结构资格与冻结任务 roster

结构筛选不读取 intervention outcome 或 proposed-method gain。六个任务都能定义稳定 phase anchor、environment-only 注入、注入瞬间无直接 cause violation、chunk 内至少剩余 8 个动作，并能进行 fresh-env prefix reconstruction；未替换任务。

| Task | Family | Structural status | Smoke config | Demo |
| --- | --- | --- | --- | ---: |
| cream-cheese → bowl | target_shift | eligible | lead08_shift040mm | 0 |
| bowl → plate | target_shift | eligible | lead08_shift100mm | 1 |
| open middle drawer | drawer_obstacle | eligible | offset-02_clear035mm | 0 |
| open top drawer + bowl inside | drawer_obstacle | eligible | offset-02_clear035mm | 0 |
| push plate → stove front | swept_path_blocker | eligible | progress070_lateral000mm | 0 |
| bowl → stove | target_region_blocker | eligible | lead08_lateral000mm | 1 |

结果：`6/6` structurally eligible，replacement_count=`0`。

## Phase 3 — 固定扰动网格资格

只使用 calibration demo 0–9；evaluation 10–39 与 held-out 40–49 均未检查。固定门槛为 immediate violation ≤10%、delayed nominal violation 30–80%、fresh replay ≥99%，并要求十个 calibration 事件 phase-valid。总计 88 个配置、880 条 calibration nominal 记录。

| Task | Qualified configs | Frozen primary | Immediate | Delayed nominal | Replay |
| --- | ---: | --- | ---: | ---: | ---: |
| cream-cheese → bowl | 10 | lead08_shift040mm | 0.000 | 0.600 | 1.000 |
| bowl → plate | 0 | none | n/a | n/a | n/a |
| open middle drawer | 0 | none | n/a | n/a | n/a |
| open top drawer + bowl inside | 0 | none | n/a | n/a | n/a |
| push plate → stove front | 0 | none | n/a | n/a | n/a |
| bowl → stove | 2 | lead14_lateral050mm | 0.000 | 0.600 | 1.000 |

通过的两任务为 cream-cheese → bowl（10 个配置，primary=`lead08_shift040mm`）和 bowl → stove（2 个配置，primary=`lead14_lateral050mm`）。另外四任务是冻结负结果：bowl/plate 在 immediate contact 与 0/100% delayed rate 间跳变；两个 drawer 任务 delayed rate 最高仅 20%/10%；push 任务的结构注入合法但 16-step suffix 内 delayed rate 为 0。未在 evaluation split 调参。

为了满足未来‘至少两种 severity’而不看方法收益，preflight 从已合格 calibration 配置中机械冻结第二种 severity：cream 使用 `lead10_shift060mm`，bowl/stove 使用 `lead14_lateral075mm`。每个任务两种 severity 各 25 events。

## Phase 4 — Strong small actor

固定 actor 为 shared multitask `HistoryConditionedStateACT`：4 帧 95-D state history、3 个已执行动作、task ID、16 个 action queries；d_model=192，6 heads，3 encoder + 3 decoder layers，FFN=768，dropout=0.10。模型参数 `3,146,503`，低于 10M；无 RGB、privileged perturbation metadata 或 world model。每任务 50 demos，seeds 7/17/29，6000 steps，batch 256，AdamW lr=3e-4 / wd=1e-4，execution horizon=1，无 sweep。

2 tasks × 2 demos × 32 states、3 steps 的 GPU smoke 成功，完整 optimizer/RNG checkpoint 与 portable checkpoint 都能写入。正式训练结果：

| Seed | Final loss | Position | Gripper | Step |
| ---: | ---: | ---: | ---: | ---: |
| 7 | 0.04231 | 0.03587 | 0.00644 | 6000 |
| 17 | 0.04051 | 0.03607 | 0.00443 | 6000 |
| 29 | 0.04291 | 0.03607 | 0.00684 | 6000 |

### Clean success / phase reach gate

每任务每 seed 10 个相同 clean init states，共 180 episodes。A 门为 clean success ∈[0.30,0.90]；B 门为 late phase reach ≥0.70 且失败主要发生在到达该 phase 后。A 或 B 任一满足即 task qualified。

| Task | Seed success 7/17/29 | Clean | Grasp/open | Lift/transport | Late phase | Qualified |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| cream-cheese → bowl | 0.2/0.5/0.6 | 0.433 | 0.600 | 0.600 | 0.667 | qualified |
| bowl → plate | 0.4/0.3/0.3 | 0.333 | 0.500 | 0.333 | 0.567 | qualified |
| open middle drawer | 0.1/0.3/0.3 | 0.233 | 0.300 | 0.300 | 0.267 | not_qualified |
| open top drawer + bowl inside | 0.0/0.2/0.2 | 0.133 | 0.367 | 0.500 | 0.300 | not_qualified |
| push plate → stove front | 0.9/0.0/0.3 | 0.400 | 1.000 | 1.000 | 0.500 | qualified |
| bowl → stove | 0.9/1.0/0.8 | 0.900 | 0.867 | 0.767 | 0.900 | qualified |

结果：`4/6` actor-qualified，**actor gate=`PASS`**。与扰动资格的交集恰好是 cream-cheese → bowl 和 bowl → stove，两任务均 actor-qualified。push 虽 aggregate=0.40，但 seed success=0.9/0.0/0.3，后续不能把它解释成稳定 actor substrate；它同时也没有合格扰动。

## Actor 机制反解（post-hoc、non-gate）

弱 reference `ChunkedBCMLP` 只看当前 95-D state、按任务独立训练，并用三层 GELU MLP 直接回归 16-action chunk；新 ACT 增加已执行动作历史、state history、task token 和 transformer query decoder，同时改为 shared multitask。两项严格 matched 的 Stage-1 horizon-1 clean 比较只能识别整个 architecture/training bundle，不能识别单一组件因果：

- bowl/plate：0.067 → 0.333，+0.267；三 seed 范围仅 0.10，但主要瓶颈仍是 lift/transport（reach 0.333）。
- middle drawer：0.033 → 0.233，+0.200；仍未过 gate，主要瓶颈是早期 grasp/articulation entry（reach 0.300）。

在冻结 checkpoint 的 teacher-forced cache rows 上逐一遮蔽输入，不重训、不参与 gate。相对 full actor 的首动作 position MAE 变化如下：

| Task | Repeat-current state | Zero action history | Cyclic task ID | Seed success range |
| --- | ---: | ---: | ---: | ---: |
| cream-cheese → bowl | +9.04% | +571.31% | +52.13% | 0.400 |
| bowl → plate | +6.36% | +548.11% | +68.65% | 0.100 |
| open middle drawer | +3.36% | +692.36% | +57.76% | 0.200 |
| open top drawer + bowl inside | +3.85% | +590.83% | +91.45% | 0.200 |
| push plate → stove front | +3.31% | +962.46% | +110.68% | 0.900 |
| bowl → stove | +9.95% | +648.19% | +90.05% | 0.200 |

六任务最大的 offline dependence 都是 executed-action history（+548% 到 +962% MAE），task ID 次之（+52% 到 +111%），简单压平 state history 的影响较小（+3% 到 +10%）。结合代码路径，这支持一个受限解释：动作历史为 shared actor 提供了当前 state 难以唯一表达的执行上下文，可能解释 matched tasks 的部分提升；task token 防止多任务动作模式混叠。它不证明因果，因为遮蔽是分布外诊断，且 ACT 与 weak MLP 同时改变了架构、共享方式和训练步数。

降低/不稳定机制也能定位但不能过度归因：两个 drawer 任务在进入早期操作 phase 前就大量失败；push 的 0.90 seed range 与对 action-history/task-ID 的极强 offline sensitivity 一致，说明其 aggregate 提升由 seed-specific trajectory mode 主导，不是稳健机制；bowl/stove 0.90 接近 gate 上界，但没有 matched weak reference，不能声称具体组件带来提升。没有生成新 idea，也没有训练 Track-B diagnostic probe。

## Phase 5 — Fresh-environment replay gate

正锚为合格 cream-cheese 配置，负锚为无合格配置的 middle drawer structural smoke。每个锚在 k=2/8/16 各用 3 个独立 fresh environment 重建；不共享 mutable env，snapshot restore 不是 primary initializer。

| Anchor | Task | Branch groups | Pass | State error | Contact/outcome | Chunk hash |
| --- | --- | ---: | ---: | ---: | --- | --- |
| positive | cream-cheese → bowl | 3 | 3/3 | 0.0 | True | True |
| negative | open middle drawer | 3 | 3/3 | 0.0 | True | True |

总体：state match `6/6`，max error `0.0`，**replay gate=`PASS`**。

## Novelty 双审边界

两名 reviewer 独立收敛：广义 policy-relative recovery/replanability window 已被 CheckVLA、prefix-Q chunking、DEHP/BCP 和 option interruption 强覆盖，不能赋 N3。仅保留窄的 oracle protocol 边界：检测后显式 target-cause S(k) 过滤、冻结 actor 的 Cπ(k)、真实 stride-1 分支，以及 interior optimum 相对 d/k_last_safe 的比较。当前标签为 **`N2_ORACLE_PROTOCOL_BOUNDARY_ONLY`**；在完整 Atlas 出现可复现 interior oracle gap 前没有算法新颖性。

## Atlas 预估工作量（尚未执行）

冻结交集为 `2` tasks × 每任务 50 events（30 evaluation + 20 held-out）× 3 actor seeds × 14 个 stride-1 prefixes（k=2…15），因此 primary R(k) 为 `4200` branch runs。三种 secondary operators 只在 d/k_best/k_last_safe 运行，最多 `2700`，总上限 `6900`；不含 source-prefix reconstruction overhead。B0–B5、A_original、N_oracle 从相同 R(k) branches 派生，不重复乘七。

完整 frozen event schedule 已生成，但状态明确为 `PREFLIGHT_ONLY_NOT_LAUNCHED`。

## 资源、恢复与测试

- 在开发机 PAI DSW 本地完成；机器有 4×A800，但本轮最多只有 1 个 GPU worker（低于计划上限 2），CPU 校准与 replay 可并行。
- 未提交 PAI DLC/spot/web-orchestrator 外部作业：本地 A800 足以完成首 checkpoint，遵守 local-first 与不做无意义 probe 的约束。
- 正式 actor 每 2000 steps 保存 model/optimizer/scheduler/RNG 完整 checkpoint，启动时扫描 complete marker 自动恢复；portable model 权重进入 Git。
- 单元/契约测试：`10 passed`，`2 warnings`；测试 XML 已归档。
- 两个长进程（扰动资格、seed29 eval）在运行期间外层 `commands.sh` 被编辑，模块已原子写完后旧 Bash 读取偏移文本导致尾部 exit 127。已按 schema、行数（880 calibration；60 seed29 episodes）和 summary 验证数据完整；该 wrapper incident 单独归档，未伪装成 clean exit。

## 当前 blocker 与下一条命令

没有 actor、replay 或基础设施 blocker。当前唯一停止条件是计划规定的 first-checkpoint barrier；同时后续范围被四个扰动负任务收窄到两任务。完整 Atlas runner 尚未运行，Track A/Track B 的统计结论仍不可得。

下一条安全、可复现命令是（重验 frozen intersection、severity/event schedule 和 6,900 branch 上限，不执行任何 branch）：

```bash
./experiments/r16_p14_stage2a/commands.sh atlas-preflight
```

本 checkpoint 报告写入并交付后，下一阶段才能在该 frozen schedule 上实现并先运行单 event/单 seed CPU branch smoke；在 smoke 的 replay/hash/预算字段通过前，不应提交完整 Atlas。

## 决策字段的严格分离

- `stage1b_universal_hypothesis`: **KILLED_IMMUTABLE**。
- `track_a_conditional_original`: **INCONCLUSIVE**（Atlas 未运行）。
- `track_b_replanability`: **INCONCLUSIVE**（Atlas 未运行）。
- diagnostic predictability: **NOT_RUN**；本轮 input masking 只是 actor 机理反解，不是 Track-B probe。
- `operator_router`: **NOT_ACTIVATED**。
- learned/deployable evidence: **none**。
- `overall`: 未赋值；最终枚举必须等待 Atlas，不能用 checkpoint PASS 冒充最终研究结论。
- `accepted=false`；full_atlas_launched=false。

## 关键证据路径

- `experiments/r16_p14_stage2a/preregistration.yaml`：冻结合同和两次 pre-calibration structural amendments。
- `artifacts/stage2a/perturbations/`：880 raw records、88-grid summary、negative results、frozen parameters。
- `artifacts/stage2a/actor/`：3 portable checkpoints、180 rollout records、qualification 与 mechanism audit。
- `artifacts/stage2a/replay/`：18 fresh-env reconstructions 与 gate summary。
- `artifacts/stage2a/atlas/`：仅 preflight schedule，未有 branch output。
- `artifacts/stage2a/SHA256SUMS`：首 checkpoint artifact hashes。
